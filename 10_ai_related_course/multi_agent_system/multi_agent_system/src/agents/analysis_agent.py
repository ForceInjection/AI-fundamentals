import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import json
import statistics
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentResult, AgentStatus, BeliefType, Desire

class AnalysisType(Enum):
    """分析类型枚举"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    COMPARATIVE = "comparative"
    TREND = "trend"
    CORRELATION = "correlation"
    ANOMALY = "anomaly"

@dataclass
class AnalysisResult:
    """分析结果数据结构"""
    analysis_type: AnalysisType
    metrics: Dict[str, float]
    insights: List[str]
    visualizations: List[Dict[str, Any]]
    confidence_score: float
    recommendations: List[str]
    timestamp: datetime
    
class AnalysisAgent(BaseAgent):
    """分析智能体 - 专门负责数据分析和洞察提取"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        
        # 分析相关配置
        self.analysis_timeout = config.get("analysis_timeout", 300)
        self.min_confidence_threshold = config.get("min_confidence_threshold", 0.6)
        self.max_data_points = config.get("max_data_points", 100000)
        
        # 分析历史和缓存
        self.analysis_cache: Dict[str, AnalysisResult] = {}
        self.cache_ttl = timedelta(hours=config.get("cache_ttl_hours", 12))
        
        # 专业分析领域
        self.analysis_domains = config.get("analysis_domains", [
            "business_intelligence", "performance_metrics", "user_behavior", 
            "market_trends", "operational_efficiency"
        ])
        
        # 初始化分析能力
        self._initialize_analysis_capabilities()
    
    def _initialize_analysis_capabilities(self):
        """初始化分析能力"""
        self.capabilities.extend([
            "statistical_analysis",
            "trend_analysis",
            "correlation_analysis",
            "anomaly_detection",
            "predictive_modeling",
            "data_visualization",
            "insight_extraction",
            "recommendation_generation"
        ])
        
        # 加载分析工具
        self.tools.update({
            "descriptive_stats": self._descriptive_statistics,
            "trend_analysis": self._trend_analysis,
            "correlation_analysis": self._correlation_analysis,
            "anomaly_detection": self._anomaly_detection,
            "comparative_analysis": self._comparative_analysis,
            "predictive_analysis": self._predictive_analysis,
            "generate_insights": self._generate_insights,
            "create_visualizations": self._create_visualizations
        })
        
        # 添加初始分析信念
        self.add_belief("min_sample_size", 30, BeliefType.FACT, 1.0, "config")
        self.add_belief("significance_level", 0.05, BeliefType.FACT, 1.0, "config")
        self.add_belief("correlation_threshold", 0.7, BeliefType.FACT, 1.0, "config")
    
    async def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """执行分析任务"""
        data = input_data.get("data")
        analysis_type = input_data.get("analysis_type", "descriptive")
        target_metrics = input_data.get("target_metrics", [])
        context = input_data.get("context", {})
        
        if data is None:
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.ERROR,
                data={},
                execution_time=0,
                error_message="No data provided for analysis"
            )
        
        try:
            # 数据预处理
            processed_data = await self._preprocess_data(data)
            
            # 执行分析
            analysis_results = await self._perform_analysis(
                processed_data, analysis_type, target_metrics, context
            )
            
            # 生成洞察和建议
            insights = await self._generate_insights(analysis_results, context)
            recommendations = await self._generate_recommendations(analysis_results, insights)
            
            # 创建可视化
            visualizations = await self._create_visualizations(analysis_results, analysis_type)
            
            # 计算置信度
            confidence_score = self._calculate_analysis_confidence(analysis_results)
            
            final_result = AnalysisResult(
                analysis_type=AnalysisType(analysis_type),
                metrics=analysis_results.get("metrics", {}),
                insights=insights,
                visualizations=visualizations,
                confidence_score=confidence_score,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
            return AgentResult(
                agent_id=self.agent_id,
                status=AgentStatus.COMPLETED,
                data={
                    "analysis_type": analysis_type,
                    "results": analysis_results,
                    "insights": insights,
                    "recommendations": recommendations,
                    "visualizations": visualizations,
                    "confidence_score": confidence_score,
                    "data_summary": self._summarize_data(processed_data)
                },
                execution_time=0,  # 将在基类中设置
                confidence_score=confidence_score
            )
            
        except Exception as e:
            self.logger.error(f"Analysis execution failed: {str(e)}")
            raise
    
    async def _preprocess_data(self, data: Union[Dict, List, pd.DataFrame]) -> pd.DataFrame:
        """数据预处理"""
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data]) if not isinstance(list(data.values())[0], list) else pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # 基本数据清理
        # 处理缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # 处理分类变量的缺失值
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('Unknown')
        
        # 数据类型转换
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass  # 保持原始类型
        
        # 限制数据点数量
        if len(df) > self.max_data_points:
            df = df.sample(n=self.max_data_points, random_state=42)
            self.logger.warning(f"Data truncated to {self.max_data_points} points")
        
        return df
    
    async def _perform_analysis(self, data: pd.DataFrame, analysis_type: str, 
                              target_metrics: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """执行具体分析"""
        analysis_type_enum = AnalysisType(analysis_type)
        
        if analysis_type_enum == AnalysisType.DESCRIPTIVE:
            return await self._descriptive_statistics(data, target_metrics)
        elif analysis_type_enum == AnalysisType.TREND:
            return await self._trend_analysis(data, target_metrics, context)
        elif analysis_type_enum == AnalysisType.CORRELATION:
            return await self._correlation_analysis(data, target_metrics)
        elif analysis_type_enum == AnalysisType.ANOMALY:
            return await self._anomaly_detection(data, target_metrics)
        elif analysis_type_enum == AnalysisType.COMPARATIVE:
            return await self._comparative_analysis(data, target_metrics, context)
        elif analysis_type_enum == AnalysisType.PREDICTIVE:
            return await self._predictive_analysis(data, target_metrics, context)
        else:
            # 默认执行描述性统计
            return await self._descriptive_statistics(data, target_metrics)
    
    async def _descriptive_statistics(self, data: pd.DataFrame, target_metrics: List[str] = None) -> Dict[str, Any]:
        """描述性统计分析"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_metrics:
            available_metrics = [col for col in target_metrics if col in numeric_data.columns]
            if available_metrics:
                numeric_data = numeric_data[available_metrics]
        
        if numeric_data.empty:
            return {"error": "No numeric data available for analysis"}
        
        stats = {
            "metrics": {},
            "summary": {},
            "distribution": {}
        }
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) == 0:
                continue
                
            stats["metrics"][column] = {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75)),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis())
            }
            
            # 分布特征
            stats["distribution"][column] = {
                "is_normal": abs(col_data.skew()) < 1 and abs(col_data.kurtosis()) < 3,
                "outliers_count": self._count_outliers(col_data),
                "unique_values": int(col_data.nunique()),
                "missing_percentage": float((len(data) - len(col_data)) / len(data) * 100)
            }
        
        # 整体数据摘要
        stats["summary"] = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "numeric_columns": len(numeric_data.columns),
            "categorical_columns": len(data.select_dtypes(include=['object']).columns),
            "missing_data_percentage": float(data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100)
        }
        
        return stats
    
    async def _trend_analysis(self, data: pd.DataFrame, target_metrics: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """趋势分析"""
        time_column = context.get("time_column", "timestamp")
        
        if time_column not in data.columns:
            # 尝试找到时间列
            time_columns = [col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_columns:
                time_column = time_columns[0]
            else:
                return {"error": "No time column found for trend analysis"}
        
        # 确保时间列是datetime类型
        try:
            data[time_column] = pd.to_datetime(data[time_column])
        except:
            return {"error": f"Cannot convert {time_column} to datetime"}
        
        # 按时间排序
        data_sorted = data.sort_values(time_column)
        
        trends = {"metrics": {}, "patterns": {}}
        
        numeric_columns = data_sorted.select_dtypes(include=[np.number]).columns
        if target_metrics:
            numeric_columns = [col for col in target_metrics if col in numeric_columns]
        
        for column in numeric_columns:
            col_data = data_sorted[[time_column, column]].dropna()
            if len(col_data) < 3:
                continue
            
            # 计算趋势指标
            values = col_data[column].values
            
            # 线性趋势
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1)
            
            # 变化率
            pct_change = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            
            # 波动性
            volatility = np.std(np.diff(values)) / np.mean(values) * 100 if np.mean(values) != 0 else 0
            
            trends["metrics"][column] = {
                "slope": float(slope),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "percentage_change": float(pct_change),
                "volatility": float(volatility),
                "start_value": float(values[0]),
                "end_value": float(values[-1]),
                "peak_value": float(np.max(values)),
                "trough_value": float(np.min(values))
            }
            
            # 模式识别
            trends["patterns"][column] = self._identify_patterns(values)
        
        return trends
    
    async def _correlation_analysis(self, data: pd.DataFrame, target_metrics: List[str] = None) -> Dict[str, Any]:
        """相关性分析"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_metrics:
            available_metrics = [col for col in target_metrics if col in numeric_data.columns]
            if available_metrics:
                numeric_data = numeric_data[available_metrics]
        
        if len(numeric_data.columns) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}
        
        # 计算相关性矩阵
        correlation_matrix = numeric_data.corr()
        
        correlations = {"metrics": {}, "strong_correlations": [], "matrix": {}}
        
        # 转换相关性矩阵为字典格式
        correlations["matrix"] = correlation_matrix.to_dict()
        
        # 找出强相关性
        correlation_threshold = self.beliefs["correlation_threshold"].value
        
        for i, col1 in enumerate(correlation_matrix.columns):
            for j, col2 in enumerate(correlation_matrix.columns):
                if i < j:  # 避免重复和自相关
                    corr_value = correlation_matrix.loc[col1, col2]
                    if not np.isnan(corr_value) and abs(corr_value) >= correlation_threshold:
                        correlations["strong_correlations"].append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr_value),
                            "strength": self._interpret_correlation_strength(abs(corr_value)),
                            "direction": "positive" if corr_value > 0 else "negative"
                        })
        
        # 计算平均相关性强度
        abs_correlations = np.abs(correlation_matrix.values)
        np.fill_diagonal(abs_correlations, 0)  # 排除对角线
        avg_correlation = np.mean(abs_correlations[abs_correlations > 0])
        
        correlations["metrics"] = {
            "average_correlation_strength": float(avg_correlation) if not np.isnan(avg_correlation) else 0,
            "strong_correlations_count": len(correlations["strong_correlations"]),
            "variables_analyzed": len(correlation_matrix.columns)
        }
        
        return correlations
    
    async def _anomaly_detection(self, data: pd.DataFrame, target_metrics: List[str] = None) -> Dict[str, Any]:
        """异常检测"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_metrics:
            available_metrics = [col for col in target_metrics if col in numeric_data.columns]
            if available_metrics:
                numeric_data = numeric_data[available_metrics]
        
        if numeric_data.empty:
            return {"error": "No numeric data available for anomaly detection"}
        
        anomalies = {"metrics": {}, "anomalous_points": {}, "summary": {}}
        
        total_anomalies = 0
        
        for column in numeric_data.columns:
            col_data = numeric_data[column].dropna()
            if len(col_data) < 10:  # 需要足够的数据点
                continue
            
            # 使用IQR方法检测异常值
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 找出异常值
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # 使用Z-score方法作为补充
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            z_outliers = col_data[z_scores > 3]
            
            anomaly_indices = list(set(outliers.index.tolist() + z_outliers.index.tolist()))
            
            anomalies["anomalous_points"][column] = {
                "indices": anomaly_indices,
                "values": [float(col_data.loc[idx]) for idx in anomaly_indices],
                "count": len(anomaly_indices),
                "percentage": float(len(anomaly_indices) / len(col_data) * 100)
            }
            
            anomalies["metrics"][column] = {
                "outlier_count": len(anomaly_indices),
                "outlier_percentage": float(len(anomaly_indices) / len(col_data) * 100),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "mean": float(col_data.mean()),
                "std": float(col_data.std())
            }
            
            total_anomalies += len(anomaly_indices)
        
        anomalies["summary"] = {
            "total_anomalies": total_anomalies,
            "columns_analyzed": len([col for col in numeric_data.columns if col in anomalies["metrics"]]),
            "overall_anomaly_rate": float(total_anomalies / len(data) * 100) if len(data) > 0 else 0
        }
        
        return anomalies
    
    async def _comparative_analysis(self, data: pd.DataFrame, target_metrics: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """对比分析"""
        group_column = context.get("group_column")
        
        if not group_column or group_column not in data.columns:
            return {"error": "Group column not specified or not found for comparative analysis"}
        
        numeric_data = data.select_dtypes(include=[np.number])
        if target_metrics:
            available_metrics = [col for col in target_metrics if col in numeric_data.columns]
            if available_metrics:
                numeric_data = numeric_data[available_metrics]
        
        if numeric_data.empty:
            return {"error": "No numeric data available for comparative analysis"}
        
        # 按组分组
        grouped = data.groupby(group_column)
        
        comparisons = {"metrics": {}, "group_stats": {}, "significant_differences": []}
        
        # 获取组信息
        groups = list(grouped.groups.keys())
        comparisons["group_stats"]["groups"] = groups
        comparisons["group_stats"]["group_sizes"] = {str(group): len(grouped.get_group(group)) for group in groups}
        
        # 对每个数值列进行组间比较
        for column in numeric_data.columns:
            group_values = {}
            group_stats = {}
            
            for group in groups:
                group_data = grouped.get_group(group)[column].dropna()
                if len(group_data) > 0:
                    group_values[str(group)] = group_data.tolist()
                    group_stats[str(group)] = {
                        "mean": float(group_data.mean()),
                        "median": float(group_data.median()),
                        "std": float(group_data.std()),
                        "count": len(group_data)
                    }
            
            comparisons["metrics"][column] = {
                "group_stats": group_stats,
                "overall_variance": self._calculate_between_group_variance(group_values),
                "coefficient_of_variation": self._calculate_cv_between_groups(group_stats)
            }
            
            # 检查显著性差异（简化版本）
            if len(group_stats) >= 2:
                means = [stats["mean"] for stats in group_stats.values()]
                if max(means) - min(means) > np.std(means):  # 简单的显著性检验
                    comparisons["significant_differences"].append({
                        "metric": column,
                        "groups_compared": list(group_stats.keys()),
                        "difference_magnitude": float(max(means) - min(means))
                    })
        
        return comparisons
    
    async def _predictive_analysis(self, data: pd.DataFrame, target_metrics: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """预测分析（简化版本）"""
        target_column = context.get("target_column")
        
        if not target_column or target_column not in data.columns:
            return {"error": "Target column not specified or not found for predictive analysis"}
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if target_column not in numeric_data.columns:
            return {"error": "Target column must be numeric for predictive analysis"}
        
        # 简单的线性趋势预测
        target_data = numeric_data[target_column].dropna()
        
        if len(target_data) < 10:
            return {"error": "Insufficient data for predictive analysis"}
        
        # 时间序列预测（假设数据按时间排序）
        x = np.arange(len(target_data))
        y = target_data.values
        
        # 线性回归
        slope, intercept = np.polyfit(x, y, 1)
        
        # 预测未来几个点
        forecast_periods = context.get("forecast_periods", 5)
        future_x = np.arange(len(target_data), len(target_data) + forecast_periods)
        predictions = slope * future_x + intercept
        
        # 计算预测置信度（基于历史拟合度）
        fitted_values = slope * x + intercept
        mse = np.mean((y - fitted_values) ** 2)
        r_squared = 1 - (np.sum((y - fitted_values) ** 2) / np.sum((y - np.mean(y)) ** 2))
        
        prediction_results = {
            "metrics": {
                "model_type": "linear_regression",
                "r_squared": float(r_squared),
                "mse": float(mse),
                "trend_slope": float(slope),
                "forecast_periods": forecast_periods
            },
            "predictions": {
                "values": [float(pred) for pred in predictions],
                "confidence_intervals": self._calculate_prediction_intervals(predictions, mse)
            },
            "model_performance": {
                "accuracy_score": float(max(0, r_squared)),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            }
        }
        
        return prediction_results
    
    def _count_outliers(self, data: pd.Series) -> int:
        """计算异常值数量"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return len(data[(data < lower_bound) | (data > upper_bound)])
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """解释相关性强度"""
        if correlation >= 0.9:
            return "very_strong"
        elif correlation >= 0.7:
            return "strong"
        elif correlation >= 0.5:
            return "moderate"
        elif correlation >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def _identify_patterns(self, values: np.ndarray) -> Dict[str, Any]:
        """识别数据模式"""
        patterns = {}
        
        # 季节性检测（简化版本）
        if len(values) >= 12:
            # 检查是否有周期性模式
            autocorr_12 = np.corrcoef(values[:-12], values[12:])[0, 1] if len(values) > 12 else 0
            patterns["seasonality_12"] = float(autocorr_12) if not np.isnan(autocorr_12) else 0
        
        # 趋势强度
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        patterns["trend_strength"] = abs(float(slope))
        
        # 波动性
        patterns["volatility"] = float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
        
        return patterns
    
    def _calculate_between_group_variance(self, group_values: Dict[str, List[float]]) -> float:
        """计算组间方差"""
        all_values = []
        group_means = []
        group_sizes = []
        
        for group_data in group_values.values():
            if group_data:
                all_values.extend(group_data)
                group_means.append(np.mean(group_data))
                group_sizes.append(len(group_data))
        
        if not group_means:
            return 0.0
        
        overall_mean = np.mean(all_values)
        between_group_variance = sum(
            size * (mean - overall_mean) ** 2 
            for mean, size in zip(group_means, group_sizes)
        ) / sum(group_sizes)
        
        return float(between_group_variance)
    
    def _calculate_cv_between_groups(self, group_stats: Dict[str, Dict[str, float]]) -> float:
        """计算组间变异系数"""
        means = [stats["mean"] for stats in group_stats.values()]
        if not means:
            return 0.0
        
        cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
        return float(cv)
    
    def _calculate_prediction_intervals(self, predictions: np.ndarray, mse: float) -> List[Dict[str, float]]:
        """计算预测区间"""
        std_error = np.sqrt(mse)
        confidence_level = 1.96  # 95% 置信区间
        
        intervals = []
        for pred in predictions:
            intervals.append({
                "lower": float(pred - confidence_level * std_error),
                "upper": float(pred + confidence_level * std_error)
            })
        
        return intervals
    
    async def _generate_insights(self, analysis_results: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        """生成分析洞察"""
        insights = []
        
        # 基于分析结果生成洞察
        if "metrics" in analysis_results:
            metrics = analysis_results["metrics"]
            
            # 描述性统计洞察
            if isinstance(metrics, dict) and any(isinstance(v, dict) for v in metrics.values()):
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and "mean" in metric_data:
                        cv = metric_data.get("std", 0) / metric_data.get("mean", 1) if metric_data.get("mean", 0) != 0 else 0
                        if cv > 1:
                            insights.append(f"{metric_name} shows high variability (CV: {cv:.2f})")
                        elif cv < 0.1:
                            insights.append(f"{metric_name} shows very stable values (CV: {cv:.2f})")
        
        # 趋势洞察
        if "patterns" in analysis_results:
            for metric_name, patterns in analysis_results["patterns"].items():
                if patterns.get("trend_strength", 0) > 0.1:
                    insights.append(f"{metric_name} shows a clear trend pattern")
                if patterns.get("volatility", 0) > 0.5:
                    insights.append(f"{metric_name} exhibits high volatility")
        
        # 相关性洞察
        if "strong_correlations" in analysis_results:
            strong_corrs = analysis_results["strong_correlations"]
            if strong_corrs:
                insights.append(f"Found {len(strong_corrs)} strong correlations between variables")
                for corr in strong_corrs[:3]:  # 只显示前3个
                    insights.append(
                        f"{corr['variable1']} and {corr['variable2']} are {corr['strength']}ly "
                        f"{corr['direction']}ly correlated ({corr['correlation']:.2f})"
                    )
        
        # 异常检测洞察
        if "summary" in analysis_results and "overall_anomaly_rate" in analysis_results["summary"]:
            anomaly_rate = analysis_results["summary"]["overall_anomaly_rate"]
            if anomaly_rate > 5:
                insights.append(f"High anomaly rate detected: {anomaly_rate:.1f}% of data points")
            elif anomaly_rate < 1:
                insights.append(f"Data quality appears good with low anomaly rate: {anomaly_rate:.1f}%")
        
        return insights[:10]  # 限制洞察数量
    
    async def _generate_recommendations(self, analysis_results: Dict[str, Any], insights: List[str]) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于洞察生成建议
        for insight in insights:
            if "high variability" in insight:
                recommendations.append("Consider investigating factors causing high variability")
            elif "stable values" in insight:
                recommendations.append("Stable metrics indicate consistent performance")
            elif "clear trend" in insight:
                recommendations.append("Monitor trend continuation and plan accordingly")
            elif "high volatility" in insight:
                recommendations.append("Implement measures to reduce volatility if needed")
            elif "strongly correlated" in insight:
                recommendations.append("Leverage strong correlations for predictive modeling")
            elif "High anomaly rate" in insight:
                recommendations.append("Investigate data quality issues and outlier causes")
        
        # 通用建议
        if not recommendations:
            recommendations.append("Continue monitoring key metrics for changes")
            recommendations.append("Consider setting up automated alerts for significant changes")
        
        return recommendations[:5]  # 限制建议数量
    
    async def _create_visualizations(self, analysis_results: Dict[str, Any], analysis_type: str) -> List[Dict[str, Any]]:
        """创建可视化配置"""
        visualizations = []
        
        if analysis_type == "descriptive":
            visualizations.extend([
                {"type": "histogram", "title": "Distribution Analysis", "data_source": "metrics"},
                {"type": "box_plot", "title": "Outlier Detection", "data_source": "distribution"}
            ])
        elif analysis_type == "trend":
            visualizations.extend([
                {"type": "line_chart", "title": "Trend Analysis", "data_source": "metrics"},
                {"type": "slope_chart", "title": "Trend Direction", "data_source": "patterns"}
            ])
        elif analysis_type == "correlation":
            visualizations.extend([
                {"type": "heatmap", "title": "Correlation Matrix", "data_source": "matrix"},
                {"type": "scatter_plot", "title": "Strong Correlations", "data_source": "strong_correlations"}
            ])
        elif analysis_type == "anomaly":
            visualizations.extend([
                {"type": "scatter_plot", "title": "Anomaly Detection", "data_source": "anomalous_points"},
                {"type": "bar_chart", "title": "Anomaly Summary", "data_source": "summary"}
            ])
        
        return visualizations
    
    def _calculate_analysis_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """计算分析置信度"""
        confidence_factors = []
        
        # 数据质量因子
        if "summary" in analysis_results:
            summary = analysis_results["summary"]
            if "missing_data_percentage" in summary:
                missing_pct = summary["missing_data_percentage"]
                data_quality_factor = max(0, 1 - missing_pct / 100)
                confidence_factors.append(data_quality_factor)
        
        # 样本大小因子
        if "summary" in analysis_results and "total_rows" in analysis_results["summary"]:
            sample_size = analysis_results["summary"]["total_rows"]
            min_sample = self.beliefs["min_sample_size"].value
            sample_factor = min(sample_size / min_sample, 1.0)
            confidence_factors.append(sample_factor)
        
        # 分析特定因子
        if "r_squared" in analysis_results.get("metrics", {}):
            # 预测分析的R²值
            r_squared = analysis_results["metrics"]["r_squared"]
            confidence_factors.append(max(0, r_squared))
        
        # 计算综合置信度
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        else:
            return 0.7  # 默认置信度
    
    def _summarize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """数据摘要"""
        return {
            "shape": list(data.shape),
            "columns": list(data.columns),
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "memory_usage": float(data.memory_usage(deep=True).sum() / 1024 / 1024),  # MB
            "null_counts": data.isnull().sum().to_dict()
        }
    
    def _generate_goals(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成分析相关的目标"""
        goals = []
        
        # 如果有待分析的数据集
        if "pending_analysis" in context:
            for dataset in context["pending_analysis"]:
                goals.append({
                    "goal_id": f"analyze_{dataset.replace(' ', '_')}",
                    "description": f"Perform comprehensive analysis on {dataset}",
                    "priority": 8,
                    "success_criteria": {"min_confidence": 0.7, "min_insights": 3}
                })
        
        return goals
    
    def _create_plan(self, goal: Desire) -> List[Dict[str, Any]]:
        """为分析目标创建执行计划"""
        if goal.goal_id.startswith("analyze_"):
            return [
                {"action": "descriptive_stats", "parameters": {"target_metrics": []}},
                {"action": "correlation_analysis", "parameters": {}},
                {"action": "anomaly_detection", "parameters": {}},
                {"action": "generate_insights", "parameters": {"min_insights": 3}}
            ]
        
        return []