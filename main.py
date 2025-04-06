from argparse import ArgumentParser 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import logging

import src.utils as utils
import src.plots as plots
from src.training_and_evaluation.loss import MSELoss
from src.models.controller import Controller
from src.dataloader.csv_dataset import CSVDataset
from src.training_and_evaluation.learner import OnlineLearner
from syne_tune import Reporter

logging = logging.getLogger('pytorch_lightning')

def main(args: ArgumentParser) -> None:
    # Set seed for reproducibility
    pl.seed_everything(args.seed)
    
    # 定义需要批量测试的病人列表，例如 1 到 100
    patient_ids = list(range(1, 101))
    for patient in patient_ids:
        print(f"开始测试病人 {patient} 的模型...")
        # 根据规则计算 site_id：
        # 如果病人编号为单数字，则 site_id = 重复该数字3次 + '3'；否则 site_id = str(patient) + '3'
        if patient < 10:
            computed_site_id = f"{str(patient)*3}3"   # 例如 patient=1 --> "1113"
        else:
            computed_site_id = f"{patient}3"           # 例如 patient=98 --> "983"
        
        # 更新 args.site_id 为计算后的 site_id
        args.site_id = computed_site_id
        
        # 生成实验名称：格式为 "病人ID-test3"，例如 "8-test3"
        experiment_name = f"{patient}-test3"
        experiment_path = os.path.join(args.save_dir, experiment_name)
        experiment_path = utils.create_experiment_folder(experiment_path, "./src")
    
        # 设置日志
        utils.config_logger(experiment_path)
        logging.info("开始病人 %s 的实验： %s", patient, experiment_name)
        logging.info("Arguments: %s", args)
    
        # 保存参数配置
        utils.save_pickle(args, utils.args_file(experiment_path))
    
        # 加载数据，CSVDataset 内部将根据 site_id 定位数据
        print(f"Using site_id: {args.site_id}")
        dataset = CSVDataset(site_id=args.site_id,
                             lookback_window=args.lookback_window,
                             prediction_horizon=args.prediction_horizon,
                             train_batchsize=args.train_batchsize,
                             valid_batchsize=args.valid_batchsize,
                             input_features=args.input_features,
                             output_features=args.output_features)
        # 每次都对加载目录进行替换，将 "1-train" 替换为 "{patient}-train"
        patient_load_dir = args.load_dir.replace("1-train", f"{patient}-train")
        logging.info("Loading dataset scaling constants from %s", patient_load_dir)
        utils.load_dataset(dataset, utils.dataset_file(patient_load_dir))
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
        # 创建控制器（模型）
        controller = Controller(dataset=dataset,
                                hidden_dim=args.hidden_dim,
                                num_layers=args.num_layers)
        logging.info("Loading model from %s", patient_load_dir)
        controller = utils.load_model(controller, utils.model_file(patient_load_dir))
    
        # 创建损失函数
        regression_loss = MSELoss()
    
        # 配置 TensorBoard 日志
        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=experiment_path, name="logs") if args.st_checkpoint_dir is None else None
    
        # 初始化 Trainer
        trainer = pl.Trainer(gpus=args.gpu, max_epochs=1, enable_checkpointing=False,
                             accelerator="auto", logger=tb_logger,
                             enable_progress_bar=True if args.st_checkpoint_dir is None else False)
    
        # 创建 Learner
        learner = OnlineLearner(controller=controller,
                                regression_loss=regression_loss,
                                dataset=dataset,
                                learning_rate=args.learning_rate,
                                weight_decay_start_weight=args.weight_decay_start_weight,
                                weight_decay_end_weight=args.weight_decay_start_weight,
                                weight_decay_start_time_step=args.weight_decay_start_time_step, 
                                weight_decay_end_time_step=args.weight_decay_end_time_step,
                                swa_gamma=args.swa_gamma,
                                swa_start_time_step=args.swa_start_time_step, 
                                swa_end_time_step=args.swa_end_time_step, 
                                swa_replace_frequency=args.swa_replace_frequency,
                                mse_start_weight=args.mse_start_weight,
                                mse_exp_decay=args.mse_exp_decay,
                                mse_end_weight=args.mse_end_weight,
                                mse_start_time_step=args.mse_start_time_step, 
                                mse_end_time_step=args.mse_end_time_step)
    
        # 训练模型
        trainer.fit(learner, train_dataloaders=dataloader)
    
        # 绘图及结果保存
        results = {}
        if args.plot:
            data_splits = ['train', 'val', 'test']
        else:
            data_splits = ['test']
    
        for data_split in data_splits:
            results[data_split] = {}
            plots.plot_predictions(experiment_path=experiment_path,
                                   dataset=dataset,
                                   predictions=learner.predictions[data_split],
                                   targets=learner.targets[data_split],
                                   output_features=args.output_features,
                                   t_initial=0, t_final=learner.predictions[data_split].shape[0],
                                   data_split=data_split)
    
        if args.plot:
            plots.plot_training_loss(experiment_path=experiment_path, training_losses=learner.training_losses)
            logging.info("Training loss plot saved at: %s", os.path.join(experiment_path, 'training_loss', 'training_loss_plot.png'))
    
        # 存储结果
        utils.store_metrics(results, metrics=learner.metrics["train"], prefix="train")
        utils.store_metrics(results, metrics=learner.metrics["val"], prefix="val")
        utils.store_metrics(results, metrics=learner.metrics["test"], prefix="test")
        logging.info("Results: %s", results)
    
        # 保存模型、结果、learner 和 dataset
        utils.save_model(controller, utils.model_file(experiment_path))
        utils.save_pickle(results, utils.results_file(experiment_path))
        utils.save_learner(learner, utils.learner_file(experiment_path))
        utils.save_learner_as_csv(learner, experiment_path)
        utils.save_dataset(dataset, utils.dataset_file(experiment_path))
        
        print(f"病人 {patient} 的测试已完成，结果保存在 {experiment_path}\n")
    
if __name__ == '__main__':
    parser = ArgumentParser()
    
    parser.add_argument_group("Data")
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='数据存放目录')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='数据加载时使用的工作线程数')
    parser.add_argument('--st_checkpoint_dir', type=str, default=None,
                        help='syne-tune 检查点存放目录')
    
    parser = CSVDataset.add_specific_args(parser)
    
    parser.add_argument_group("Network")
    parser = Controller.add_specific_args(parser)
    
    parser.add_argument_group("Experiment")
    parser.add_argument('--seed', type=int, default=42,
                        help='训练随机种子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='使用的 GPU 编号')
    parser.add_argument('--save_dir', type=str, default='experiments',
                        help='实验结果保存目录')
    parser.add_argument('--load_dir', type=str, default="/home/zceelub/BG_Recursive_complete_version/experiments/1-train/",
                        help='模型加载目录，格式应为 "/.../1-train/"，将自动替换为对应病人编号')
    parser.add_argument('--plot', type=int, choices=[0, 1], default=1,
                        help='是否绘制训练和验证结果图')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='实验目录名称')
    
    parser = OnlineLearner.add_specific_args(parser)
    args = parser.parse_args()
    
    main(args)
