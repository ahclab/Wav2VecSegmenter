from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor

upperchar_vocab = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-large-960h-lv60-self"
).tokenizer.get_vocab()


class BaseVocabulary:
    """Base class for Vocabulary"""

    def __init__(self):
        self.word2id = {
            "<B>": 0,
            "<NB>": 1,
            "<PAD>": 2,
            "<SEP>": 3,
        }
        self.n_special_tokens = len(self.word2id)
        self.set_properties()

    def set_properties(self):
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.boundary_token = self.id2word[0]
        self.boundary_token_id = self.word2id["<B>"]
        self.nonboundary_token = self.id2word[1]
        self.nonboundary_token_id = self.word2id["<NB>"]
        self.pad_token = self.id2word[2]
        self.pad_token_id = self.word2id["<PAD>"]
        self.sep_token = self.id2word[3]
        self.sep_token_id = self.word2id["<SEP>"]
        self.vocab_size = len(self.word2id)

    def get_vocab(self):
        return self.word2id


class UppercasedCharVocabulary(BaseVocabulary):
    """Uppercased character vocabulary"""

    def __init__(self):
        super().__init__()

        for k in upperchar_vocab.keys():
            upperchar_vocab[k] += self.n_special_tokens
        self.word2id = dict(**self.word2id, **upperchar_vocab)

        self.set_properties()

    def get_vocab(self):
        return self.word2id


class CollateFn:
    def __init__(self, pad_token_id) -> None:
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        batch: list,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.BoolTensor,
        list[bool],
        list[int],
        list[int],
    ]:
        """
        (inference) collate function for the dataloader of the SegmentationDataset
        Args:
            batch (list): list of examples from SegmentationDataset
        Returns:
            Tuple[ torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.BoolTensor, list[bool], list[int], list[int], ]:
                0: 2D tensor, padded and normalized waveforms for each random segment
                1: 2D tensor, binary padded targets for each random segment (output space)
                2: 2D tensor, binary mask for wav2vec 2.0 (input space)
                3: 2D tensor, binary mask for audio-frame-classifier (output space)
                4: a '0' indicates that the whole example is empty (torch.zeros)
                5: the start frames of the segments (output space)
                6: the end frames of the segments (output space)
        """

        included = [bool(example[0].sum()) for example in batch]
        starts = [example[2] for example in batch]
        ends = [example[3] for example in batch]

        # sequence lengths
        in_seq_len = [len(example[0]) for example in batch]
        out_seq_len = [end - start for start, end in zip(starts, ends)]
        bs = len(in_seq_len)

        # pad and concat
        audio = torch.cat(
            [
                F.pad(example[0], (0, max(in_seq_len) - len(example[0]))).unsqueeze(0)
                for example in batch
            ]
        )

        # check if the batch contains also targets
        if batch[0][1] is not None:
            target = torch.cat(
                [
                    F.pad(
                        example[1],
                        (0, max(out_seq_len) - len(example[1])),
                        value=self.pad_token_id,
                    ).unsqueeze(0)
                    for example in batch
                ]
            )
        else:
            target = None

        # normalize input
        # only for inputs that have non-zero elements
        included_ = torch.tensor(included).bool()
        audio[included_] = (
            audio[included_] - torch.mean(audio[included_], dim=1, keepdim=True)
        ) / torch.std(audio[included_], dim=1, keepdim=True)

        # get masks
        in_mask = torch.ones(audio.shape, dtype=torch.long)
        out_mask = torch.ones([bs, max(out_seq_len)], dtype=torch.bool)
        for i, in_sl, out_sl in zip(range(bs), in_seq_len, out_seq_len):
            in_mask[i, in_sl:] = 0
            out_mask[i, out_sl:] = 0

        return {
            "audio": audio,
            "target": target,
            "in_mask": in_mask,
            "out_mask": out_mask,
            "included": included,
            "starts": starts,
            "ends": ends,
        }


class AutoRegCollateFn:
    def __init__(self, pad_token_id) -> None:
        self.pad_token_id = pad_token_id

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def __call__(
        self,
        batch: list,
    ) -> Tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.LongTensor,
        torch.BoolTensor,
        list[bool],
        list[int],
        list[int],
    ]:
        """
        (inference) collate function for the dataloader of the SegmentationDataset
        Args:
            batch (list): list of examples from SegmentationDataset
        Returns:
            Tuple[ torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.BoolTensor, list[bool], list[int], list[int], ]:
                0: 2D tensor, padded and normalized waveforms for each random segment
                1: 2D tensor, binary padded targets for each random segment (output space)
                2: 2D tensor, binary mask for wav2vec 2.0 (input space)
                3: 2D tensor, binary mask for audio-frame-classifier (output space)
                4: a '0' indicates that the whole example is empty (torch.zeros)
                5: the start frames of the segments (output space)
                6: the end frames of the segments (output space)
        """

        included = [bool(example[0].sum()) for example in batch]
        starts = [example[2] for example in batch]
        ends = [example[3] for example in batch]

        # sequence lengths
        in_seq_len = [len(example[0]) for example in batch]
        src_seq_len = [end - start for start, end in zip(starts, ends)]
        tgt_seq_len = [len(example[1]) for example in batch]
        bs = len(in_seq_len)

        # pad and concat
        audio = torch.cat(
            [
                F.pad(example[0], (0, max(in_seq_len) - len(example[0]))).unsqueeze(0)
                for example in batch
            ]
        )

        # check if the batch contains also targets
        if batch[0][1] is not None:
            target = torch.cat(
                [
                    F.pad(
                        example[1],
                        (0, max(tgt_seq_len) - len(example[1])),
                        value=self.pad_token_id,
                    ).unsqueeze(0)
                    for example in batch
                ]
            )
        else:
            target = None

        # normalize input
        # only for inputs that have non-zero elements
        included_ = torch.tensor(included).bool()
        audio[included_] = (
            audio[included_] - torch.mean(audio[included_], dim=1, keepdim=True)
        ) / torch.std(audio[included_], dim=1, keepdim=True)

        # get masks
        in_mask = torch.ones(audio.shape, dtype=torch.long)
        tgt_pad_mask = torch.ones(
            [bs, max(tgt_seq_len) - 1], dtype=torch.bool
        )  # -1 for tail SEP
        for i, in_sl, out_sl in zip(range(bs), in_seq_len, tgt_seq_len):
            in_mask[i, in_sl:] = 0
            tgt_pad_mask[i, out_sl - 1 :] = 0  # -1 for tail SEP
        src_pad_mask = tgt_pad_mask[:, 1:]  # -1 for head SEP

        tgt_mask = self.generate_square_subsequent_mask(max(tgt_seq_len) - 1)

        return {
            "audio": audio,
            "in_target": target[:, :-1],
            "out_target": target[:, 1:],
            "in_mask": in_mask,
            "src_pad_mask": src_pad_mask,
            "tgt_pad_mask": tgt_pad_mask,
            "tgt_mask": tgt_mask,
            "included": included,
            "starts": starts,
            "ends": ends,
        }
