"""SpotSound — fine-grained audio temporal grounding on top of Audio Flamingo 3.

Port of the official implementation (https://github.com/LoieSun/SpotSound,
`model/af3.py` + `processor/af3.py`) onto the `transformers` release this Space
pins (5.9.x), keeping the two ingredients that make SpotSound work:

1. `AudioFlamingo3TemporalProcessor` — instead of expanding the `<sound>`
   placeholder into `N` plain audio tokens, it interleaves an explicit textual
   timestamp every 25 audio tokens (AF3 emits exactly 25 post-pool audio tokens
   per second of audio), i.e.

       "timestamp: 0 seconds; feature: <sound>×25timestamp: 1 seconds; feature: <sound>×25…"

   This is what lets the LM read absolute time off the audio stream.

2. `AudioFlamingo3ForTemporalConditionalGeneration` — a `forward` that scatters
   the audio features into the audio-token slots *without* the strict
   "#tokens == #features" check of the base class, because the interleaving
   above deliberately keeps only `floor(N / 25) * 25` of the `N` audio features.
"""

import re

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AudioFlamingo3Processor

from spantool import SPANTOOL_PROMPT, timestamped_audio_expansion

# AF3 emits 25 post-pool audio tokens per second of audio.
TOKENS_PER_SECOND = 25

# The exact prompts SpotSound was trained with (see upstream `train.py`).
GROUNDING_PROMPT = (
    "This is a sequence of audio stream. Your task is to identify the temporal "
    "window (start and end timestamps) when the given query appears. The query is: "
)
DETECTION_PROMPT = (
    "This is a sequence of audio stream. Your task is to identify whether the "
    "sound event in the query occurs. The query is: "
)
class AudioFlamingo3TemporalProcessor(AudioFlamingo3Processor):
    """AF3 processor with SpotSound's timestamp-interleaved audio-token expansion."""

    def _expand_audio_tokens(self, text, padding_mask, per_sample_windows):
        audio_lengths = torch.stack(
            [s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)]
        )
        audio_tokens_lengths = self._get_audio_token_length(audio_lengths)
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for i, num_audio_tokens in enumerate(audio_tokens_lengths):
            audio_duration = int(num_audio_tokens) // TOKENS_PER_SECOND
            expanded = "".join(
                f"timestamp: {t} seconds; feature: " + self.audio_token * TOKENS_PER_SECOND
                for t in range(audio_duration)
            )
            text[i] = audio_token_pattern.sub(lambda _m: expanded, text[i], count=1)
        return text


class AudioFlamingo3SpanToolProcessor(AudioFlamingo3TemporalProcessor):
    """Query-first processor that preserves the final partial second of audio.

    The official temporal processor expands only complete groups of 25 audio
    tokens.  SpanTool consumes frame states directly, so dropping the remainder
    would make the end of every non-integer-duration clip unobservable.
    """

    def _expand_audio_tokens(self, text, padding_mask, per_sample_windows):
        audio_lengths = torch.stack(
            [s.sum() for s in torch.split(padding_mask.sum(-1), per_sample_windows)]
        )
        audio_tokens_lengths = self._get_audio_token_length(audio_lengths)
        audio_token_pattern = re.compile(re.escape(self.audio_token))
        for index, num_audio_tokens in enumerate(audio_tokens_lengths):
            expanded = timestamped_audio_expansion(
                self.audio_token, int(num_audio_tokens), TOKENS_PER_SECOND
            )
            text[index] = audio_token_pattern.sub(
                lambda _match: expanded, text[index], count=1
            )
        return text


class AudioFlamingo3ForTemporalConditionalGeneration(AudioFlamingo3ForConditionalGeneration):
    """AF3 generation model whose audio scatter tolerates truncated feature counts."""

    def forward(
        self,
        input_ids=None,
        input_features=None,
        input_features_mask=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        logits_to_keep=0,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None and input_ids is not None:
            audio_embeds = self.get_audio_features(
                input_features, input_features_mask, return_dict=True
            ).pooler_output

            # Replace audio-token placeholders with audio embeddings, in order.
            audio_token_mask = input_ids == self.config.audio_token_id
            n_slots = int(audio_token_mask.sum())
            n_feats = int(audio_embeds.shape[0])
            if n_slots > n_feats:  # defensive: only fill the slots we have features for
                flat = audio_token_mask.reshape(-1).clone()
                flat[flat.nonzero(as_tuple=True)[0][n_feats:]] = False
                audio_token_mask = flat.reshape(input_ids.shape)
                n_slots = n_feats
            audio_embeds = audio_embeds[:n_slots].to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(
                audio_token_mask.unsqueeze(-1).to(inputs_embeds.device), audio_embeds
            )

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


def build_conversation(audio, query: str, prompt: str = GROUNDING_PROMPT):
    """Build the single-turn chat SpotSound was trained on.

    `audio` is either a path (str) or a 16 kHz mono numpy array.
    """
    audio_content = (
        {"type": "audio", "path": audio}
        if isinstance(audio, str)
        else {"type": "audio", "audio": audio}
    )
    return [
        {
            "role": "user",
            "content": [
                audio_content,
                {"type": "text", "text": prompt + query + " Answer: "},
            ],
        }
    ]


def build_spantool_conversation(audio, query: str):
    """Put the query before audio so causal audio states are query-conditioned."""
    audio_content = (
        {"type": "audio", "path": audio}
        if isinstance(audio, str)
        else {"type": "audio", "audio": audio}
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": SPANTOOL_PROMPT + query + "\nAudio: "},
                audio_content,
            ],
        }
    ]
