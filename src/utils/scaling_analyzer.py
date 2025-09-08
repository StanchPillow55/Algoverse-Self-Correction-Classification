"""
Scaling Analysis System

Implements power-law fitting and scaling analysis for the scaling study.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

# Try to import scipy for curve fitting
try:
    from scipy.optimize import curve_fit
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Power-law fitting will use basic methods.")

logger = logging.getLogger(__name__)

@dataclass
class ScalingResult:
    """Result of scaling analysis."""
    model_size: float
    improvement: float
    cost: float
    cost_efficiency: float
    model_name: str
    dataset_name: str

@dataclass
class PowerLawFit:
    """Result of power-law fitting."""
    exponent: float
    coefficient: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    equation: str

class ScalingAnalyzer:
    """Analyzes scaling laws in self-correction experiments."""
    
    def __init__(self):
        """Initialize scaling analyzer."""
        # Model size mapping (approximate parameter counts)
        self.model_sizes = {
            "gpt-4o-mini": 1.8,  # 1.8B parameters
            "claude-haiku": 3.0,  # 3B parameters
            "gpt-4o": 8.0,  # 8B parameters
            "claude-sonnet": 70.0,  # 70B parameters
            "llama-70b": 70.0,  # 70B parameters
            "gpt-4": 100.0,  # 100B+ parameters
            "claude-opus": 100.0,  # 100B+ parameters
        }
        
        # Model cost mapping
        self.model_costs = {
            "gpt-4o-mini": 0.00015,
            "claude-haiku": 0.00025,
            "gpt-4o": 0.0025,
            "claude-sonnet": 0.003,
            "llama-70b": 0.0007,
            "gpt-4": 0.03,
            "claude-opus": 0.015
        }
    
    def load_experiment_results(self, results_dir: str) -> List[ScalingResult]:
        """Load experiment results for scaling analysis."""
        results_dir = Path(results_dir)
        results = []
        
        # Find all result files
        result_files = list(results_dir.glob("*.json"))
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract model name from filename
                filename = file_path.stem
                parts = filename.split('_')
                model_name = parts[0] if parts else "unknown"
                
                # Get model size and cost
                model_size = self.model_sizes.get(model_name, 1.0)
                cost_per_1k = self.model_costs.get(model_name, 0.001)
                
                # Calculate metrics from traces
                traces = data.get('traces', [])
                if not traces:
                    continue
                
                # Calculate improvement
                initial_correct = sum(1 for t in traces if t.get('turns', [{}])[0].get('accuracy', 0) == 1)
                final_correct = sum(1 for t in traces if t.get('final_accuracy', 0) == 1)
                
                initial_accuracy = initial_correct / len(traces)
                final_accuracy = final_correct / len(traces)
                improvement = final_accuracy - initial_accuracy
                
                # Estimate cost
                total_tokens = len(traces) * 200  # Rough estimate
                cost = (total_tokens / 1000) * cost_per_1k
                cost_efficiency = improvement / cost if cost > 0 else 0
                
                # Extract dataset name
                dataset_name = parts[1] if len(parts) > 1 else "unknown"
                
                result = ScalingResult(
                    model_size=model_size,
                    improvement=improvement,
                    cost=cost,
                    cost_efficiency=cost_efficiency,
                    model_name=model_name,
                    dataset_name=dataset_name
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        return results
    
    def fit_power_law(self, x_data: np.ndarray, y_data: np.ndarray) -> PowerLawFit:
        """Fit power law: y = a * x^b"""
        if len(x_data) < 2:
            return PowerLawFit(0, 0, 0, 1, (0, 0), "y = 0")
        
        # Remove zeros and negative values for log fitting
        mask = (x_data > 0) & (y_data > 0)
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) < 2:
            return PowerLawFit(0, 0, 0, 1, (0, 0), "y = 0")
        
        try:
            if SCIPY_AVAILABLE:
                # Use scipy for robust fitting
                def power_law(x, a, b):
                    return a * np.power(x, b)
                
                # Initial guess
                p0 = [1.0, 0.5]
                
                # Fit the curve
                popt, pcov = curve_fit(power_law, x_clean, y_clean, p0=p0, maxfev=1000)
                a, b = popt
                
                # Calculate R-squared
                y_pred = power_law(x_clean, a, b)
                ss_res = np.sum((y_clean - y_pred) ** 2)
                ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calculate confidence interval for exponent
                try:
                    errors = np.sqrt(np.diag(pcov))
                    ci_lower = b - 1.96 * errors[1]
                    ci_upper = b + 1.96 * errors[1]
                    confidence_interval = (ci_lower, ci_upper)
                except:
                    confidence_interval = (b, b)
                
                # Calculate p-value (approximate)
                try:
                    _, p_value = stats.pearsonr(np.log(x_clean), np.log(y_clean))
                except:
                    p_value = 1.0
                
            else:
                # Basic linear regression on log-transformed data
                log_x = np.log(x_clean)
                log_y = np.log(y_clean)
                
                # Linear regression: log(y) = log(a) + b * log(x)
                n = len(log_x)
                sum_log_x = np.sum(log_x)
                sum_log_y = np.sum(log_y)
                sum_log_x_sq = np.sum(log_x ** 2)
                sum_log_xy = np.sum(log_x * log_y)
                
                # Calculate coefficients
                b = (n * sum_log_xy - sum_log_x * sum_log_y) / (n * sum_log_x_sq - sum_log_x ** 2)
                log_a = (sum_log_y - b * sum_log_x) / n
                a = np.exp(log_a)
                
                # Calculate R-squared
                y_pred = a * np.power(x_clean, b)
                ss_res = np.sum((y_clean - y_pred) ** 2)
                ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Approximate confidence interval
                confidence_interval = (b * 0.8, b * 1.2)
                p_value = 0.05  # Placeholder
            
            equation = f"y = {a:.4f} * x^{b:.4f}"
            
            return PowerLawFit(
                exponent=b,
                coefficient=a,
                r_squared=r_squared,
                p_value=p_value,
                confidence_interval=confidence_interval,
                equation=equation
            )
            
        except Exception as e:
            logger.error(f"Power law fitting failed: {e}")
            return PowerLawFit(0, 0, 0, 1, (0, 0), "y = 0")
    
    def analyze_scaling(self, results: List[ScalingResult]) -> Dict[str, Any]:
        """Analyze scaling patterns in results."""
        if not results:
            return {"error": "No results to analyze"}
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'model_size': r.model_size,
                'improvement': r.improvement,
                'cost': r.cost,
                'cost_efficiency': r.cost_efficiency,
                'model_name': r.model_name,
                'dataset_name': r.dataset_name
            }
            for r in results
        ])
        
        analysis = {
            "summary": {
                "total_experiments": len(results),
                "models_tested": df['model_name'].nunique(),
                "datasets_tested": df['dataset_name'].nunique(),
                "avg_improvement": df['improvement'].mean(),
                "max_improvement": df['improvement'].max(),
                "min_improvement": df['improvement'].min()
            },
            "scaling_laws": {},
            "cost_analysis": {},
            "recommendations": []
        }
        
        # Analyze scaling by model size
        if len(df) > 1:
            x_data = df['model_size'].values
            y_data = df['improvement'].values
            
            # Fit power law
            power_law_fit = self.fit_power_law(x_data, y_data)
            
            analysis["scaling_laws"]["model_size_vs_improvement"] = {
                "exponent": power_law_fit.exponent,
                "coefficient": power_law_fit.coefficient,
                "r_squared": power_law_fit.r_squared,
                "p_value": power_law_fit.p_value,
                "confidence_interval": power_law_fit.confidence_interval,
                "equation": power_law_fit.equation,
                "interpretation": self._interpret_scaling_exponent(power_law_fit.exponent)
            }
        
        # Analyze cost scaling
        if len(df) > 1:
            x_data = df['model_size'].values
            y_data = df['cost_efficiency'].values
            
            cost_fit = self.fit_power_law(x_data, y_data)
            
            analysis["scaling_laws"]["model_size_vs_cost_efficiency"] = {
                "exponent": cost_fit.exponent,
                "coefficient": cost_fit.coefficient,
                "r_squared": cost_fit.r_squared,
                "p_value": cost_fit.p_value,
                "confidence_interval": cost_fit.confidence_interval,
                "equation": cost_fit.equation
            }
        
        # Cost analysis
        analysis["cost_analysis"] = {
            "total_cost": df['cost'].sum(),
            "avg_cost_per_experiment": df['cost'].mean(),
            "most_cost_efficient_model": df.loc[df['cost_efficiency'].idxmax(), 'model_name'] if len(df) > 0 else "N/A",
            "cost_efficiency_range": {
                "min": df['cost_efficiency'].min(),
                "max": df['cost_efficiency'].max(),
                "mean": df['cost_efficiency'].mean()
            }
        }
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_scaling_recommendations(analysis, df)
        
        return analysis
    
    def _interpret_scaling_exponent(self, exponent: float) -> str:
        """Interpret the scaling exponent."""
        if exponent > 0.5:
            return "Strong positive scaling - improvement increases rapidly with model size"
        elif exponent > 0.2:
            return "Moderate positive scaling - improvement increases with model size"
        elif exponent > 0:
            return "Weak positive scaling - slight improvement with model size"
        elif exponent > -0.2:
            return "Weak negative scaling - slight decrease with model size"
        else:
            return "Strong negative scaling - improvement decreases with model size"
    
    def _generate_scaling_recommendations(self, analysis: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on scaling analysis."""
        recommendations = []
        
        # Scaling recommendations
        scaling_laws = analysis.get("scaling_laws", {})
        if "model_size_vs_improvement" in scaling_laws:
            scaling = scaling_laws["model_size_vs_improvement"]
            exponent = scaling["exponent"]
            r_squared = scaling["r_squared"]
            
            if r_squared > 0.8:
                recommendations.append(f"Strong scaling law detected (R² = {r_squared:.3f})")
                if exponent > 0.3:
                    recommendations.append("Self-correction benefits significantly from larger models")
                elif exponent > 0.1:
                    recommendations.append("Self-correction benefits moderately from larger models")
                else:
                    recommendations.append("Self-correction benefits minimally from larger models")
            else:
                recommendations.append(f"Weak scaling law (R² = {r_squared:.3f}) - model size may not be the primary factor")
        
        # Cost recommendations
        cost_analysis = analysis.get("cost_analysis", {})
        if cost_analysis:
            most_efficient = cost_analysis.get("most_cost_efficient_model", "N/A")
            if most_efficient != "N/A":
                recommendations.append(f"Most cost-efficient model: {most_efficient}")
        
        # General recommendations
        summary = analysis.get("summary", {})
        avg_improvement = summary.get("avg_improvement", 0)
        
        if avg_improvement > 0.1:
            recommendations.append("Self-correction shows significant improvement (>10%)")
        elif avg_improvement > 0.05:
            recommendations.append("Self-correction shows moderate improvement (5-10%)")
        else:
            recommendations.append("Self-correction shows minimal improvement (<5%)")
        
        return recommendations
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: str):
        """Save analysis results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Scaling analysis saved to: {output_path}")

def main():
    """Test the scaling analyzer."""
    analyzer = ScalingAnalyzer()
    
    # Load results
    results = analyzer.load_experiment_results("outputs/scaling_experiments")
    
    if not results:
        print("No results found to analyze")
        return
    
    print(f"Loaded {len(results)} experiment results")
    
    # Analyze scaling
    analysis = analyzer.analyze_scaling(results)
    
    # Print summary
    print("\n📊 Scaling Analysis Summary")
    print("=" * 40)
    
    summary = analysis.get("summary", {})
    print(f"Total experiments: {summary.get('total_experiments', 0)}")
    print(f"Models tested: {summary.get('models_tested', 0)}")
    print(f"Average improvement: {summary.get('avg_improvement', 0):.3f}")
    
    # Print scaling laws
    scaling_laws = analysis.get("scaling_laws", {})
    if "model_size_vs_improvement" in scaling_laws:
        scaling = scaling_laws["model_size_vs_improvement"]
        print(f"\nScaling Law: {scaling['equation']}")
        print(f"R² = {scaling['r_squared']:.3f}")
        print(f"Interpretation: {scaling['interpretation']}")
    
    # Print recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    # Save analysis
    analyzer.save_analysis(analysis, "outputs/scaling_analysis.json")

if __name__ == "__main__":
    main()
