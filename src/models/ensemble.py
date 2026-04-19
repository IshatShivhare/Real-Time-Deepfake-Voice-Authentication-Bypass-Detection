import torch
import torch.nn.functional as F
import torchaudio.functional as TAF
import numpy as np

class Ensemble:
    def __init__(self, models: dict, wav2vec2_weight: float = 0.5,
                 rawnet2_weight: float = 0.5, threshold: float = 0.5):
        self.models = models
        self.wav2vec2_weight = wav2vec2_weight
        self.rawnet2_weight = rawnet2_weight
        self.threshold = threshold

    def predict(self, audio_array: np.ndarray, sample_rate: int) -> dict:
        target_sr = 16000
        max_samples = 64000
        
        # Resample if necessary
        audio_tensor = torch.from_numpy(audio_array).float()
        if sample_rate != target_sr:
            audio_tensor = TAF.resample(audio_tensor, sample_rate, target_sr)
            
        # Clip or pad to exactly max_samples
        if audio_tensor.shape[0] > max_samples:
            audio_tensor = audio_tensor[:max_samples]
        elif audio_tensor.shape[0] < max_samples:
            padding = max_samples - audio_tensor.shape[0]
            audio_tensor = F.pad(audio_tensor, (0, padding))
            
        wav2vec2_model = self.models["wav2vec2"]
        wav2vec2_processor = self.models["wav2vec2_processor"]
        rawnet2_model = self.models["rawnet2"]
        
        with torch.no_grad():
            # Wav2Vec2 inference
            input_values = wav2vec2_processor(audio_tensor.numpy(), sampling_rate=target_sr, return_tensors="pt").input_values
            logit, attn = wav2vec2_model(input_values)
            wav2vec2_score = torch.sigmoid(logit).item()
            
            # RawNet2 inference
            rn_input = audio_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, T]
            rn_logits = rawnet2_model(rn_input)
            rawnet2_score = F.softmax(rn_logits, dim=1)[:, 1].item()
            
        final_score = self.wav2vec2_weight * wav2vec2_score + self.rawnet2_weight * rawnet2_score
        verdict = "SPOOF" if final_score > self.threshold else "REAL"
        confidence = abs(final_score - 0.5) * 2
        
        return {
            "final_score": final_score,
            "verdict": verdict,
            "wav2vec2_score": wav2vec2_score,
            "rawnet2_score": rawnet2_score,
            "confidence": confidence,
            "attn_weights": attn.numpy() if attn is not None else None
        }
