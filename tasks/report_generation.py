import torch
from transformers import AutoModel

class Prism:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = AutoModel.from_pretrained(
            "/home/cjt/project_script/coca_pytorch/prism",
            trust_remote_code=True
        ).to(device)
        print("🚀 加载Prism成功")

    def Report_generation(self, feature):
        embedding_data = torch.load(feature)
        tile_embeddings = embedding_data['embeddings'].unsqueeze(0).to(self.device)

        with torch.autocast(self.device, torch.float16), torch.inference_mode():
            reprs = self.model.slide_representations(tile_embeddings)

        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            genned_ids = self.model.generate(
                key_value_states=reprs['image_latents'],
                do_sample=False,
                num_beams=5,
                num_beam_groups=1,
            )
            genned_caption = self.model.untokenize(genned_ids)
        return genned_caption