import yaml
import os
from data.dataset_manager import DatasetManager
from models.model_manager import ModelConfigManager
from preprocessing.trident_wrapper import TridentRunner
from tasks import report_generation
import json
from utils.metrics import Metrics

if __name__ == "__main__":
    # 1. 加载配置文件
    with open("./configs/config.yaml") as f:
        config = yaml.safe_load(f)

    # 2. 初始化数据管理器
    dataset_mgr = DatasetManager(config["data"]["index_files"])
    dataset_name = config["data"]["data_sets"]
    dataset = dataset_mgr.get_dataset_paths(dataset_name)
    dataset_path = dataset["slides"]
    label = dataset["label"]

    # 3. 初始化测试模型
    model_mgr = ModelConfigManager("configs/models/models.yaml")
    model_name = config["model"]["name"]
    model = model_mgr.get_model_config(model_name)

    # 4. 预处理数据
    preprocess_root_dir = config["data"]["preprocessed_dir"]
    preprocess_dir = os.path.join(preprocess_root_dir, dataset_name, model_name)
    trident = TridentRunner()
    trident.run(
        wsi_dir = dataset_path,
        job_dir = preprocess_dir,
        patch_encoder = model["patch_encoder"],
        mag = model["mag"],
        patch_size = model["patch_size"],
        overlap = model["overlap"],
        gpu = config["gpu"]["id"]
    )

    # 5. 下游任务运行以及结果保存
    task_name = config["task"]["name"]

    ########## 创建结果文件 ################
    reslut_root_dir = config["result"]["result_dir"]
    reslut_dir = os.path.join(reslut_root_dir, dataset_name, model_name, task_name, "result.json")

    ########## 加载模型 ################

    gpu = config["gpu"]["id"]
    feat_dir = os.path.join(preprocess_dir, f"{model["mag"]}x_{model["patch_size"]}px_{model["overlap"]}px_overlap", f"feature_{model["patch_encoder"]}")
    if task_name == "Report_generation":
        if model_name == "Prism":
            task = report_generation.Prism(device=f"cuda:{gpu}")
            ########### 遍历推理 ################
            with open(reslut_dir, "w", encoding="utf-8") as f:
                for slide in os.listdir(feat_dir):
                    reslut = task.Report_generation(slide)
                    reslut_dump = {
                        "slide": f"{reslut}"
                    }
                    json.dump(reslut_dump, f, ensure_ascii=False, indent=2)


    # 6. 结果输出图表
    eval_name = config["eval"]["name"]
    eval = Metrics()
    if eval_name  == "BLEU":
        eval.bleu(label, reslut_dir)
        