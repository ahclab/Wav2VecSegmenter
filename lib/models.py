import math

import torch
from constants import HIDDEN_SIZE
from hydra.utils import instantiate
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2BaseModelOutput


class AutoRegSegmenter(nn.Module):
    def __init__(
        self,
        wav2vec_model_name,
        wav2vec_keep_layers,
        finetune_wav2vec,
        wav2vec_ft_layers,
        finetune_w2v_feat_enc,
        n_transformer_enc_layers,
        n_transformer_enc_heads,
        n_transformer_dec_layers,
        n_transformer_dec_heads,
        init_dropout,
        vocab_size,
    ) -> None:
        super().__init__()

        self.wav2vec_model = HFWav2Vec2(
            wav2vec_model_name,
            wav2vec_keep_layers,
            finetune_wav2vec,
            wav2vec_ft_layers,
            finetune_w2v_feat_enc,
        )

        self.seg_model = TransformerEncoderDecoder(
            HIDDEN_SIZE,
            n_transformer_enc_layers,
            n_transformer_enc_heads,
            n_transformer_dec_layers,
            n_transformer_dec_heads,
            init_dropout,
            vocab_size,
        )

    def forward(
        self,
        audio,
        target,
        in_mask,
        src_pad_mask,
        tgt_pad_mask,
        tgt_mask,
    ):
        h = self.wav2vec_model(audio, in_mask)
        output = self.seg_model(h, target, src_pad_mask, tgt_pad_mask, tgt_mask)

        return output


class TransformerEncoderDecoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_transformer_enc_layers: int = 1,
        n_transformer_enc_heads: int = 8,
        n_transformer_dec_layers: int = 4,
        n_transformer_dec_heads: int = 8,
        init_dropout: float = 0.1,
        vocab_size: int = 2,
    ) -> None:
        super().__init__()

        if n_transformer_enc_layers:
            self.transformer_encoder = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model,
                    nhead=n_transformer_enc_heads,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_transformer_enc_layers,
            )

        if n_transformer_dec_layers:
            self.transformer_decoder = torch.nn.TransformerDecoder(
                torch.nn.TransformerDecoderLayer(
                    d_model,
                    nhead=n_transformer_dec_heads,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_transformer_dec_layers,
            )
            self.positional_encoding = PositionalEncoding(d_model, dropout=init_dropout)
            self.tgt_tok_emb = TokenEmbedding(vocab_size, d_model)

        self.dropout = torch.nn.Dropout(p=init_dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.output_layer = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self,
        src: torch.FloatTensor,
        tgt: torch.FloatTensor,
        src_key_padding_mask: torch.LongTensor,
        tgt_key_padding_mask: torch.LongTensor,
        tgt_mask: torch.FloatTensor,
    ) -> torch.FloatTensor:
        src_key_padding_mask = ~src_key_padding_mask.bool()
        tgt_key_padding_mask = ~tgt_key_padding_mask.bool()

        x = self.dropout(src)

        if hasattr(self, "transformer_encoder"):
            x = self.transformer_encoder(
                x,
                src_key_padding_mask=src_key_padding_mask,
            )
            x = self.layer_norm(x)

        if hasattr(self, "transformer_decoder"):
            tgt_emb = self.tgt_tok_emb(tgt)
            # [TODO] PE
            # [TODO] tgt_emb = self.positional_encoding(tgt_emb)
            out = self.transformer_decoder(
                tgt_emb,
                x,
                tgt_mask,
                None,
                tgt_key_padding_mask,
                src_key_padding_mask,
            )

        logits = self.output_layer(self.layer_norm(out))

        return logits.squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.embedding_size = embedding_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.embedding_size)


class SHAS(nn.Module):
    def __init__(
        self,
        wav2vec_model_name,
        wav2vec_keep_layers,
        finetune_wav2vec,
        wav2vec_ft_layers,
        finetune_w2v_feat_enc,
        finetune_w2v_ffn,
        ffn_adapter,
        n_transformer_enc_layers,
        n_transformer_enc_heads,
        init_dropout,
    ) -> None:
        super().__init__()

        if finetune_wav2vec and ffn_adapter:
            self.wav2vec_model = HFWav2Vec2WithAdapter(
                wav2vec_model_name,
                wav2vec_keep_layers,
                finetune_wav2vec,
                wav2vec_ft_layers,
                finetune_w2v_feat_enc,
                finetune_w2v_ffn,
            )
        else:
            self.wav2vec_model = HFWav2Vec2(
                wav2vec_model_name,
                wav2vec_keep_layers,
                finetune_wav2vec,
                wav2vec_ft_layers,
                finetune_w2v_feat_enc,
                finetune_w2v_ffn,
            )

        self.seg_model = SegmentationFrameClassifier(
            HIDDEN_SIZE,
            n_transformer_enc_layers,
            n_transformer_enc_heads,
            init_dropout,
        )

    def forward(
        self,
        audio,
        in_mask,
        out_mask,
    ):
        _, h = self.wav2vec_model(audio, in_mask)

        # some times the output of wav2vec is 1 frame larger/smaller
        # correct for these cases
        size1 = h.shape[1]
        size2 = out_mask.shape[1]
        if size1 != size2:
            if size1 < size2:
                out_mask = out_mask[:, :-1]
                # RM ends = [e - 1 for e in ends]
            else:
                h = h[:, :-1, :]

        output = self.seg_model(h, out_mask)

        return output


class SHASWithSSL(nn.Module):
    def __init__(
        self,
        wav2vec_model_name,
        finetune_wav2vec,
        wav2vec_ft_layers,
        finetune_w2v_feat_enc,
        n_transformer_enc_layers,
        n_transformer_enc_heads,
        init_dropout,
        vocab_size,
    ) -> None:
        super().__init__()

        self.wav2vec_model = HFWav2Vec2ForCTC(
            wav2vec_model_name,
            finetune_wav2vec,
            wav2vec_ft_layers,
            finetune_w2v_feat_enc,
        )

        self.seg_model = SegmentationFrameClassifier(
            HIDDEN_SIZE,
            n_transformer_enc_layers,
            n_transformer_enc_heads,
            init_dropout,
            vocab_size,
        )

    def forward(
        self,
        audio,
        in_mask,
        out_mask,
    ):
        target_ctc, h = self.wav2vec_model(audio, in_mask)
        output = self.seg_model(h, out_mask)

        return [target_ctc, output]


class SegmentationFrameClassifier(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_transformer_enc_layers: int = 1,
        n_transformer_enc_heads: int = 8,
        init_dropout: float = 0.1,
        vocab_size: int = 1,
    ) -> None:
        super().__init__()

        if n_transformer_enc_layers:
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model,
                    nhead=n_transformer_enc_heads,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=n_transformer_enc_layers,
            )

        self.dropout = torch.nn.Dropout(p=init_dropout)
        self.layer_norm = torch.nn.LayerNorm(d_model)

        self.output_layer = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.FloatTensor, attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        attention_mask = ~attention_mask.bool()

        x = self.dropout(x)

        if hasattr(self, "transformer"):
            x = self.transformer(x, src_key_padding_mask=attention_mask)

        logits = self.output_layer(self.layer_norm(x))

        return logits.squeeze(-1)


class HFWav2Vec2(nn.Module):
    def __init__(
        self,
        wav2vec_model_name,
        wav2vec_keep_layers,
        finetune: bool = False,
        wav2vec_ft_layers: int = None,
        finetune_feature_encoder: bool = True,
        finetune_feed_forward: bool = True,
    ):
        super().__init__()

        self.model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

        # keep only some layers of wav2vec2
        self.model.encoder.layers = torch.nn.modules.container.ModuleList(
            [
                layer
                for i, layer in enumerate(self.model.encoder.layers)
                if i < wav2vec_keep_layers
            ]
        )
        # also remove final layer norm since it corresponds to the 24th-layer
        # the input of the classifier will be normalized again
        self.model.encoder.layer_norm = torch.nn.Identity()

        if finetune:
            if not finetune_feature_encoder:
                for layer in self.model.feature_extractor.conv_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
                for p in self.model.feature_projection.parameters():
                    p.requires_grad = False
            for i, layer in enumerate(self.model.encoder.layers):
                if i < (wav2vec_keep_layers - wav2vec_ft_layers):
                    for p in layer.parameters():
                        p.requires_grad = False
                    p.requires_grad = False
                if not finetune_feed_forward:
                    for p in layer.feed_forward.parameters():
                        p.requires_grad = False

    def forward(self, audio, attention_mask):
        return None, self.model(audio, attention_mask).last_hidden_state


class ScaledParallelAdapter(nn.Module):
    def __init__(self, embed_dim: int, bottleneck_dim: int, scaling_factor: float = 1):
        super().__init__()

        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.scaling_factor = scaling_factor

        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.relu = nn.ReLU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)

    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
        # x is the input to the layer that we are modifying
        # y is the output of the later that we are modifying
        adapter_out = self.up_proj(self.relu(self.down_proj(x)))
        return y + self.scaling_factor * adapter_out


class Wav2Vec2EncoderLayerStableLayerNormFFNAdapter(nn.Module):
    def __init__(self, attention, dropout, layer_norm, feed_forward, final_layer_norm):
        super().__init__()

        self.attention = attention
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.feed_forward = feed_forward
        self.final_layer_norm = final_layer_norm

        self.ffn_adapter = ScaledParallelAdapter(
            self.attention.out_proj.out_features, 512, 4
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states

        ffn_residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        ffn_out = self.feed_forward(hidden_states)
        hidden_states = self.ffn_adapter(hidden_states, ffn_out)

        hidden_states = ffn_residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class HFWav2Vec2WithAdapter(nn.Module):
    def __init__(
        self,
        wav2vec_model_name,
        wav2vec_keep_layers,
        finetune: bool = False,
        wav2vec_ft_layers: int = None,
        finetune_feature_encoder: bool = True,
        finetune_feed_forward: bool = True,
    ):
        super().__init__()

        self.model = Wav2Vec2Model.from_pretrained(wav2vec_model_name)

        enc_list = []
        for i, layer in enumerate(self.model.encoder.layers):
            if i < (wav2vec_keep_layers - wav2vec_ft_layers):
                enc_list.append(layer)
            elif i < wav2vec_keep_layers:
                enc_list.append(
                    Wav2Vec2EncoderLayerStableLayerNormFFNAdapter(
                        layer.attention,
                        layer.dropout,
                        layer.layer_norm,
                        layer.feed_forward,
                        layer.final_layer_norm,
                    )
                )
            else:
                break
        self.model.encoder.layers = torch.nn.modules.container.ModuleList(enc_list)

        self.model.encoder.layer_norm = torch.nn.Identity()

        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False
        elif finetune:
            if not finetune_feature_encoder:
                for layer in self.model.feature_extractor.conv_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
                for p in self.model.feature_projection.parameters():
                    p.requires_grad = False
            for i, layer in enumerate(self.model.encoder.layers):
                if i < (wav2vec_keep_layers - wav2vec_ft_layers):
                    for p in layer.parameters():
                        p.requires_grad = False
                    p.requires_grad = False
                if not finetune_feed_forward:
                    for p in layer.feed_forward.parameters():
                        p.requires_grad = False

    def forward(self, audio, attention_mask):
        return None, self.model(audio, attention_mask).last_hidden_state


class HFWav2Vec2ForCTC(nn.Module):
    def __init__(
        self,
        wav2vec_model_name,
        finetune: bool = False,
        wav2vec_ft_layers: int = None,  # [TODO]
        finetune_feature_encoder: bool = True,  # [TODO]
        finetune_feed_forward: bool = True,  # [TODO]
    ):
        super().__init__()

        self.model = Wav2Vec2ForCTC.from_pretrained(wav2vec_model_name)
        if not finetune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, audio, attention_mask):
        # return logits of CTC sequence and last hidden state
        s = self.model(audio, attention_mask, output_hidden_states=True)
        return s.logits, s.hidden_states[-1]
