import subprocess

class TridentRunner:
    def __init__(self, script_path="/home/cjt/project_script/trident/run_batch_of_slides.py"):
        self.script_path = script_path

    def run(self, wsi_dir, job_dir, patch_encoder, mag, patch_size, overlap, gpu):
        cmd_seg = [
            "python", self.script_path,
            "--task", "seg",
            "--wsi_dir", wsi_dir,
            "--job_dir", job_dir,
            "--gpu", gpu,
            "--segmenter", "hest"
        ]
        # print(f"[INFO] Running Trident: {' '.join(cmd_seg)}")
        subprocess.run(cmd_seg, check=True)

        cmd_patch = [
            "python", self.script_path,
            "--task", "coords",
            "--wsi_dir", wsi_dir,
            "--job_dir", job_dir,
            "--mag", str(mag),
            "--patch_size", str(patch_size),
            "--overlap", str(overlap)
        ]
        # print(f"[INFO] Running Trident: {' '.join(cmd_patch)}")
        subprocess.run(cmd_patch, check=True)

        cmd_feat = [
            "python", self.script_path,
            "--task", "feat",
            "--wsi_dir", wsi_dir,
            "--job_dir", job_dir,
            "--mag", str(mag),
            "--patch_size", str(patch_size),
            "--gpu", str(gpu),
            "--patch_encoder", patch_encoder
        ]
        # print(f"[INFO] Running Trident: {' '.join(cmd_feat)}")
        subprocess.run(cmd_feat, check=True)
