import os
from typing import Any, Dict
import shap
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from src.analyzer.analyzerBase import AnalyzerBase
from src.factory.analyzer_factory import AnalyzerFactory


# ==========================================================
# ✅ 自定义时序 LIME 类（支持 3D 输入）
# ==========================================================
class TimeSeriesLIME:
    """
    自定义时序版 LIME，可解释 (samples, timesteps, features) 的输入模型。
    原理：通过随机扰动 mask，拟合局部线性回归估计每个时间步-特征的重要性。
    """

    def __init__(self, training_data, feature_names, mode="regression", num_samples=100):
        """
        Args:
            training_data: np.ndarray, shape = (N, T, F)
            feature_names: list[str]
            mode: "regression" or "classification"
            num_samples: 每个样本生成的扰动样本数量
        """
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.num_samples = num_samples
        self.T = training_data.shape[1]
        self.F = training_data.shape[2]

    def explain_instance(self, data_row, predict_fn):
        """
        解释一个样本（时序输入）
        Args:
            data_row: shape (T, F)
            predict_fn: 函数，输入 np.ndarray(shape=(N,T,F)) 输出预测值 np.ndarray(N,)
        Returns:
            DataFrame: 每个时间步、特征的重要性权重
        """
        # 1️⃣ 随机扰动 mask (0/1)
        masks = np.random.randint(0, 2, size=(self.num_samples, self.T, self.F))

        # 2️⃣ 基于 mask 生成扰动样本
        mean_background = np.mean(self.training_data, axis=0)
        perturbed = np.array([
            data_row * mask + mean_background * (1 - mask)
            for mask in masks
        ])

        # 3️⃣ 模型预测
        preds = predict_fn(perturbed)
        preds = preds.reshape(-1)

        # 4️⃣ 展开 mask 作为特征矩阵
        X_flat = masks.reshape(self.num_samples, -1)

        # 5️⃣ 使用 Ridge 拟合局部线性模型
        reg = Ridge(alpha=1.0)
        reg.fit(X_flat, preds)
        weights = reg.coef_.reshape(self.T, self.F)

        # 6️⃣ 整理结果
        results = []
        for t in range(self.T):
            for f_idx, feat in enumerate(self.feature_names):
                results.append({
                    "Time_Step": t,
                    "Feature": feat,
                    "Weight": weights[t, f_idx]
                })
        return pd.DataFrame(results)


# ==========================================================
# ✅ 主分析器类（先 LIME 再 SHAP）
# ==========================================================
@AnalyzerFactory.register('shap_lime_analyzer')
class ShapLimeAnalyzer(AnalyzerBase):
    """
    分析器先计算 时序版 LIME，再计算 SHAP。
    支持输入 shape = (samples, timesteps, features)。
    结果保存为 CSV 文件。
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config)
        self.model_name = ""
        self.error_threshold = kwargs.get(
            "error_threshold", (config or {}).get("error_threshold", 0.1)
        )
        self.res = {}

    def analyze(
        self,
        predictions: Dict[str, Any],
        true_values: Any,
        model=None,
        X_test=None,
        feature_names=None,
        **kwargs,
    ):
        """
        执行 LIME + SHAP 分析，添加预测正确性和标签。
        """
        if model is None or X_test is None or feature_names is None:
            raise ValueError("需要提供 model, X_test 和 feature_names")

        device = next(model.parameters()).device

        # --- 转换输入 ---
        X_test_np = X_test.values if hasattr(X_test, "values") else X_test
        X_test_tensor = torch.from_numpy(X_test_np).float().to(device)

        # --- 获取预测结果和误差 ---
        first_model_key = next(iter(predictions.keys()))
        predicted_values = predictions[first_model_key][:, 0]
        true_values = np.array(true_values).reshape(-1)
        predicted_values = np.array(predicted_values).reshape(-1)
        relative_error = np.abs(
            (predicted_values - true_values) / (true_values + 1e-10)
        )

        # ==================================================
        # ✅ 1. 时序 LIME 分析
        # ==================================================
        print("[INFO] Running Time-Series LIME analysis...")

        lime_explainer = TimeSeriesLIME(
            training_data=np.array(X_test_np),
            feature_names=feature_names,
            num_samples=100
        )

        lime_results = []
        for i in range(len(X_test_np)):
            lime_df_single = lime_explainer.explain_instance(
                data_row=X_test_np[i],
                predict_fn=lambda x: model(
                    torch.from_numpy(x).float().to(device)
                ).detach().cpu().numpy(),
            )
            lime_df_single["Sample_Index"] = i
            true_val = true_values[i].item() if np.ndim(true_values[i]) > 0 else true_values[i]
            pred_val = predicted_values[i].item() if np.ndim(predicted_values[i]) > 0 else predicted_values[i]
            lime_df_single["True_Label"] = true_val
            lime_df_single["Predicted_Label"] = pred_val
            lime_df_single["Prediction_Correct"] = int(relative_error[i] <= self.error_threshold)
            lime_results.append(lime_df_single)

        lime_df = pd.concat(lime_results, ignore_index=True)
        self.res["lime"] = lime_df

        # ==================================================
        # ✅ 2. SHAP 分析
        # ==================================================
        print("[INFO] Running SHAP analysis...")

        model.train()
        try:
            explainer = shap.GradientExplainer(model, X_test_tensor)
            shap_values = explainer.shap_values(X_test_tensor)
        except Exception as e:
            torch.backends.cudnn.enabled = False
            explainer = shap.GradientExplainer(model, X_test_tensor)
            shap_values = explainer.shap_values(X_test_tensor)
        finally:
            torch.backends.cudnn.enabled = True
            model.eval()

        shap_values_avg = np.mean(shap_values, axis=1).squeeze()
        shap_values_df = pd.DataFrame(shap_values_avg, columns=feature_names)
        shap_values_df["True_Label"] = true_values
        shap_values_df["Predicted_Label"] = predicted_values
        shap_values_df["Prediction_Correct"] = (relative_error <= self.error_threshold).astype(int)

        self.res["shap"] = shap_values_df

    # ==================================================
    # ✅ 保存分析结果
    # ==================================================
    def _save_to_file(self, path: str, model_name: str) -> None:
        """
        保存 SHAP 和 LIME 分析结果为 CSV 文件。
        """
        if not self.res:
            raise ValueError("没有结果可保存")

        os.makedirs(path, exist_ok=True)

        lime_file = os.path.join(path, f"lime_values.csv")
        self.res["lime"].to_csv(lime_file, index=False)

        shap_file = os.path.join(path, f"shap_values.csv")
        self.res["shap"].to_csv(shap_file, index=False)

        print(f"✅ LIME 和 SHAP 结果已保存到: {path}")
