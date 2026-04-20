import os
import json
import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from transformers import Wav2Vec2Processor
from src.models.custom.wav2vec2_classifier import Wav2Vec2Classifier
from src.models.vocoder.rawnet2_sincnet import RawNet2WithSincNet

# Load environment variables
load_dotenv()

def load_all_models(config: dict) -> dict:
    # HuggingFace Login if token exists
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print("Authenticating with HuggingFace...")
        login(token=hf_token)
    else:
        print("No HF_TOKEN found in .env. Proceeding without login.")

    models = {}
    
    # Wav2Vec2
    w2v_cfg = config['models']['wav2vec2']
    w2v_repo = w2v_cfg['hf_repo']
    
    # Download or get cached paths
    try:
        w2v_config_path = hf_hub_download(repo_id=w2v_repo, filename=w2v_cfg['config_file'])
        w2v_weight_path = hf_hub_download(repo_id=w2v_repo, filename=w2v_cfg['weight_file'])
        
        # Cache locally
        os.makedirs(os.path.dirname(w2v_cfg['local_cache']), exist_ok=True)
        if not os.path.exists(w2v_cfg['local_cache']):
            import shutil
            shutil.copy2(w2v_weight_path, w2v_cfg['local_cache'])
            
    except Exception as e:
        print(f"Failed to download Wav2Vec2 from HuggingFace: {e}")
        if os.path.exists(w2v_cfg['local_cache']):
            print("Falling back to local cache.")
            w2v_weight_path = w2v_cfg['local_cache']
            # We assume config.json is available or we could hardcode defaults
            raise RuntimeError("HuggingFace unreachable and config not cached.")
        else:
            raise RuntimeError("Cannot load Wav2Vec2. No cache and HF failed.")
            
    with open(w2v_config_path, 'r') as f:
        w2v_arch_cfg = json.load(f)
        
    wav2vec2_model = Wav2Vec2Classifier(
        wav2vec2_name=w2v_arch_cfg.get('wav2vec2_name', 'facebook/wav2vec2-base'),
        hidden_dim=w2v_arch_cfg['hidden_dim'],
        attn_dim=w2v_arch_cfg['attn_dim'],
        dropout=w2v_arch_cfg['dropout']
    )
    wav2vec2_model.load_state_dict(torch.load(w2v_weight_path, map_location="cpu"))
    wav2vec2_model.eval()
    
    models["wav2vec2"] = wav2vec2_model
    models["wav2vec2_processor"] = Wav2Vec2Processor.from_pretrained(w2v_arch_cfg.get('wav2vec2_name', 'facebook/wav2vec2-base'))
    
    # RawNet2
    rn_cfg = config['models']['rawnet2']
    rn_repo = rn_cfg['hf_repo']
    
    try:
        rn_config_path = hf_hub_download(repo_id=rn_repo, filename=rn_cfg['config_file'])
        rn_weight_path = hf_hub_download(repo_id=rn_repo, filename=rn_cfg['weight_file'])
        
        os.makedirs(os.path.dirname(rn_cfg['local_cache']), exist_ok=True)
        if not os.path.exists(rn_cfg['local_cache']):
            import shutil
            shutil.copy2(rn_weight_path, rn_cfg['local_cache'])
    except Exception as e:
        print(f"Failed to download RawNet2 from HuggingFace: {e}")
        if os.path.exists(rn_cfg['local_cache']):
            print("Falling back to local cache.")
            rn_weight_path = rn_cfg['local_cache']
            raise RuntimeError("HuggingFace unreachable and config not cached.")
        else:
            raise RuntimeError("Cannot load RawNet2. No cache and HF failed.")
            
    with open(rn_config_path, 'r') as f:
        rn_arch_cfg = json.load(f)
        
    rawnet2_model = RawNet2WithSincNet(
        sinc_num_filters=rn_arch_cfg['sinc_num_filters'],
        sinc_kernel_size=rn_arch_cfg['sinc_kernel_size'],
        sinc_min_low_hz=rn_arch_cfg['sinc_min_low_hz'],
        sinc_min_band_hz=rn_arch_cfg['sinc_min_band_hz'],
        filts=rn_arch_cfg['rawnet2_filters'],
        gru_node=rn_arch_cfg['rawnet2_gru_nodes'],
        nb_gru_layer=rn_arch_cfg['rawnet2_gru_layers'],
        nb_fc_node=rn_arch_cfg['rawnet2_nb_fc_node'],
        nb_classes=rn_arch_cfg['rawnet2_nb_classes'],
        sample_rate=rn_arch_cfg['sample_rate']
    )
    rawnet2_model.load_state_dict(torch.load(rn_weight_path, map_location="cpu"))
    rawnet2_model.eval()
    
    models["rawnet2"] = rawnet2_model
    
    return models
