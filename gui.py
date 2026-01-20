"""
gui.py - Compact Professional GUI for Heart Disease Prediction
All features visible in a well-organized layout
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import threading
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from train import HeartDataProcessor, AdvancedHeartDiseaseANN
import torch
import joblib

class HeartDiseasePredictor:
    """Predictor that loads model correctly"""
    
    def __init__(self, model_path='heart_disease_model.pth', processor_path='data_processor.pkl'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = joblib.load(processor_path)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = AdvancedHeartDiseaseANN(input_dim=checkpoint['input_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.severity_labels = {0: "Mild", 1: "Moderate", 2: "Severe"}
        self.risk_interpretation = {
            'low': (0, 30, "Low Risk", "Minimal intervention needed"),
            'medium': (30, 70, "Medium Risk", "Lifestyle changes recommended"),
            'high': (70, 100, "High Risk", "Immediate medical consultation advised")
        }
    
    def predict_single(self, patient_data):
        input_tensor = self.processor.preprocess_single(patient_data)
        input_tensor = torch.FloatTensor(input_tensor).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        disease_prob = outputs['disease_prob'].item()
        disease_pred = 1 if disease_prob > 0.5 else 0
        severity_probs = outputs['severity'].cpu().numpy()[0]
        severity = np.argmax(severity_probs)
        risk_score = outputs['risk_score'].item()
        confidence = outputs['confidence'].item() * 100
        
        risk_category = self._get_risk_category(risk_score)
        feature_importance = self._analyze_feature_importance(patient_data)
        
        return {
            'prediction': disease_pred,
            'probability': disease_prob,
            'severity': self.severity_labels[severity],
            'severity_probs': severity_probs.tolist(),
            'risk_score': risk_score,
            'risk_category': risk_category,
            'confidence': confidence,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_risk_category(self, risk_score):
        for category, (low, high, label, advice) in self.risk_interpretation.items():
            if low <= risk_score < high:
                return {'category': category, 'label': label, 'advice': advice}
        return {'category': 'unknown', 'label': "Unknown", 'advice': "Consult healthcare provider"}
    
    def _analyze_feature_importance(self, patient_data):
        importance = {}
        for feature, value in patient_data.items():
            if feature in ['age', 'chol', 'trestbps', 'oldpeak']:
                importance[feature] = min(value / 100, 1.0)
            elif feature == 'thalach':
                importance[feature] = 1 - min(value / 200, 1.0)
            else:
                importance[feature] = abs(value)
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])

class HeartDiseaseGUI:
    """Compact professional GUI - all features visible"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Heart Disease Prediction System")
        self.root.geometry("1400x800")
        
        # Professional color scheme
        self.colors = {
            'bg': '#f5f6fa',
            'panel': '#ffffff',
            'primary': '#3498db',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'text': '#2c3e50',
            'text_light': '#7f8c8d'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Initialize predictor
        self.predictor = None
        self.initialize_predictor()
        
        self.current_result = None
        
        # Setup styles
        self.setup_styles()
        
        # Create GUI
        self.create_gui()
    
    def initialize_predictor(self):
        try:
            self.predictor = HeartDiseasePredictor()
            return True
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{str(e)}\n\nPlease run train.py first.")
            return False
    
    def setup_styles(self):
        """Setup ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'), foreground=self.colors['primary'])
        style.configure('Heading.TLabel', font=('Segoe UI', 11, 'bold'), foreground=self.colors['text'])
        style.configure('Info.TLabel', font=('Segoe UI', 9), foreground=self.colors['text'])
        style.configure('Card.TFrame', background='white', relief='raised')
    
    def create_gui(self):
        """Create compact GUI layout"""
        
        # Header
        header = tk.Frame(self.root, bg=self.colors['primary'], height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="❤️ Heart Disease Prediction System", 
                font=('Segoe UI', 16, 'bold'),
                fg='white', bg=self.colors['primary']).pack(expand=True)
        
        # Main container
        main = tk.Frame(self.root, bg=self.colors['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure grid
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=2)
        
        # LEFT: Input Panel
        self.create_input_panel(main)
        
        # RIGHT: Results Panel
        self.create_results_panel(main)
        
        # Bottom: Action buttons and status
        self.create_bottom_panel(self.root)
    
    def create_input_panel(self, parent):
        """Compact input panel with all fields visible"""
        
        panel = tk.LabelFrame(parent, text="Patient Information", 
                             font=('Segoe UI', 10, 'bold'),
                             bg=self.colors['panel'], fg=self.colors['text'],
                             padx=10, pady=10)
        panel.grid(row=0, column=0, sticky='nsew', padx=(0, 5))
        
        # Input fields with compact layout
        self.input_vars = {}
        fields = [
            ("Age:", "age", 20, 100, 50),
            ("Sex (1=M, 0=F):", "sex", 0, 1, 1),
            ("Chest Pain (0-3):", "cp", 0, 3, 0),
            ("Rest BP:", "trestbps", 90, 200, 120),
            ("Cholesterol:", "chol", 100, 600, 200),
            ("FBS >120:", "fbs", 0, 1, 0),
            ("Rest ECG:", "restecg", 0, 2, 0),
            ("Max HR:", "thalach", 60, 220, 150),
            ("Ex Angina:", "exang", 0, 1, 0),
            ("ST Dep:", "oldpeak", 0.0, 6.2, 1.0),
            ("Slope:", "slope", 0, 2, 1),
            ("Vessels:", "ca", 0, 3, 0),
            ("Thal:", "thal", 1, 3, 2)
        ]
        
        for i, (label, key, min_val, max_val, default) in enumerate(fields):
            row = i
            
            tk.Label(panel, text=label, font=('Segoe UI', 9),
                    bg=self.colors['panel'], fg=self.colors['text'],
                    anchor='w').grid(row=row, column=0, sticky='w', pady=2, padx=(0, 5))
            
            var = tk.DoubleVar(value=default)
            self.input_vars[key] = var
            
            if key == 'oldpeak':
                entry = ttk.Entry(panel, textvariable=var, width=12, font=('Segoe UI', 9))
                entry.grid(row=row, column=1, sticky='ew', pady=2)
            else:
                spinbox = ttk.Spinbox(panel, from_=min_val, to=max_val,
                                     textvariable=var, width=12, font=('Segoe UI', 9))
                spinbox.grid(row=row, column=1, sticky='ew', pady=2)
        
        panel.grid_columnconfigure(1, weight=1)
    
    def create_results_panel(self, parent):
        """Compact results panel with tabs"""
        
        panel = tk.LabelFrame(parent, text="Prediction Results",
                             font=('Segoe UI', 10, 'bold'),
                             bg=self.colors['panel'], fg=self.colors['text'],
                             padx=10, pady=10)
        panel.grid(row=0, column=1, sticky='nsew', padx=(5, 0))
        
        # Notebook for tabs
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Summary
        summary_tab = tk.Frame(notebook, bg='white')
        notebook.add(summary_tab, text='Summary')
        self.create_summary_tab(summary_tab)
        
        # Tab 2: Details
        details_tab = tk.Frame(notebook, bg='white')
        notebook.add(details_tab, text='Details')
        self.create_details_tab(details_tab)
        
        # Tab 3: Charts
        charts_tab = tk.Frame(notebook, bg='white')
        notebook.add(charts_tab, text='Charts')
        self.create_charts_tab(charts_tab)
    
    def create_summary_tab(self, parent):
        """Summary results tab"""
        
        # Result label
        self.result_label = tk.Label(parent, text="⏳ Awaiting Prediction",
                                    font=('Segoe UI', 14, 'bold'),
                                    fg=self.colors['text_light'], bg='white')
        self.result_label.pack(pady=15)
        
        # Metrics frame
        metrics = tk.Frame(parent, bg='white')
        metrics.pack(fill=tk.X, padx=20, pady=10)
        
        for i, (label, var_name, color) in enumerate([
            ("Risk Score", "risk_score", self.colors['warning']),
            ("Probability", "probability", self.colors['primary']),
            ("Confidence", "confidence", self.colors['success'])
        ]):
            card = tk.Frame(metrics, bg='#f8f9fa', relief=tk.RAISED, bd=1)
            card.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
            
            tk.Label(card, text=label, font=('Segoe UI', 9),
                    fg=self.colors['text'], bg='#f8f9fa').pack(pady=(8, 2))
            
            val_label = tk.Label(card, text="--", font=('Segoe UI', 18, 'bold'),
                                fg=color, bg='#f8f9fa')
            val_label.pack(pady=(0, 8))
            
            setattr(self, f'{var_name}_label', val_label)
        
        # Severity
        tk.Label(parent, text="Severity:", font=('Segoe UI', 10, 'bold'),
                fg=self.colors['text'], bg='white').pack(pady=(10, 5))
        
        self.severity_label = tk.Label(parent, text="--", font=('Segoe UI', 12, 'bold'),
                                      fg=self.colors['text'], bg='white')
        self.severity_label.pack(pady=(0, 10))
        
        # Recommendation
        tk.Label(parent, text="Recommendation:", font=('Segoe UI', 10, 'bold'),
                fg=self.colors['text'], bg='white').pack(pady=(10, 5))
        
        self.recommendation_text = tk.Text(parent, height=5, font=('Segoe UI', 9),
                                          bg='#fffacd', fg=self.colors['text'],
                                          wrap=tk.WORD, relief=tk.FLAT, padx=10, pady=10)
        self.recommendation_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        self.recommendation_text.insert(1.0, "Enter patient data and click 'Predict' to see results.")
        self.recommendation_text.config(state=tk.DISABLED)
    
    def create_details_tab(self, parent):
        """Details tab"""
        
        self.details_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD,
                                                      font=('Consolas', 9),
                                                      bg='white', relief=tk.FLAT)
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.details_text.insert(1.0, "Detailed analysis will appear here after prediction.")
        self.details_text.config(state=tk.DISABLED)
    
    def create_charts_tab(self, parent):
        """Charts tab"""
        
        self.fig = Figure(figsize=(7, 5), dpi=80)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Charts will appear after prediction',
               ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        self.canvas.draw()
    
    def create_bottom_panel(self, parent):
        """Bottom action panel"""
        
        bottom = tk.Frame(parent, bg=self.colors['bg'], height=60)
        bottom.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Buttons frame
        btn_frame = tk.Frame(bottom, bg=self.colors['bg'])
        btn_frame.pack(side=tk.LEFT, fill=tk.Y, pady=5)
        
        buttons = [
            ("🔮 Predict", self.predict_current, self.colors['success']),
            ("🔄 Reset", self.reset_form, self.colors['warning']),
            ("💾 Save", self.save_input, self.colors['primary']),
            ("🆘 Help", self.show_help, self.colors['text']),
            ("🚪 Exit", self.root.quit, self.colors['danger'])
        ]
        
        for text, cmd, color in buttons:
            tk.Button(btn_frame, text=text, command=cmd,
                     font=('Segoe UI', 10, 'bold'),
                     fg='white', bg=color,
                     relief=tk.FLAT, padx=15, pady=8,
                     cursor='hand2').pack(side=tk.LEFT, padx=3)
        
        # Status
        self.status_var = tk.StringVar(value="✅ Ready")
        tk.Label(bottom, textvariable=self.status_var,
                font=('Segoe UI', 9),
                fg=self.colors['text'], bg=self.colors['bg'],
                anchor='w').pack(side=tk.RIGHT, fill=tk.Y, padx=10)
    
    def predict_current(self):
        """Make prediction"""
        if not self.predictor:
            messagebox.showerror("Error", "Model not loaded.")
            return
        
        try:
            patient_data = {}
            for key, var in self.input_vars.items():
                value = var.get()
                patient_data[key] = float(value) if key in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] else int(float(value))
            
            self.status_var.set("🔍 Analyzing...")
            self.root.update()
            
            threading.Thread(target=self._predict_thread, args=(patient_data,), daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _predict_thread(self, patient_data):
        try:
            result = self.predictor.predict_single(patient_data)
            self.current_result = result
            self.root.after(0, self._update_results, result)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
    
    def _update_results(self, result):
        """Update results"""
        
        # Update summary
        if result['prediction']:
            self.result_label.config(text="⚠️ Heart Disease Detected", fg=self.colors['danger'])
        else:
            self.result_label.config(text="✅ No Heart Disease", fg=self.colors['success'])
        
        self.risk_score_label.config(text=f"{result['risk_score']:.1f}%")
        self.probability_label.config(text=f"{result['probability']*100:.1f}%")
        self.confidence_label.config(text=f"{result['confidence']:.1f}%")
        self.severity_label.config(text=result['severity'])
        
        self.recommendation_text.config(state=tk.NORMAL)
        self.recommendation_text.delete(1.0, tk.END)
        self.recommendation_text.insert(1.0, result['risk_category']['advice'])
        self.recommendation_text.config(state=tk.DISABLED)
        
        # Update details
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        
        details = f"""PREDICTION RESULTS
{'='*50}

Prediction: {'Disease Detected' if result['prediction'] else 'No Disease'}
Probability: {result['probability']:.1%}
Risk Score: {result['risk_score']:.1f}%
Severity: {result['severity']}
Confidence: {result['confidence']:.1f}%

Risk Category: {result['risk_category']['label']}
Advice: {result['risk_category']['advice']}

TOP RISK FACTORS:
"""
        for feature, score in result['feature_importance'].items():
            details += f"  • {feature}: {score:.3f}\n"
        
        self.details_text.insert(1.0, details)
        self.details_text.config(state=tk.DISABLED)
        
        # Update charts
        self.update_charts(result)
        
        self.status_var.set(f"✅ Complete - Risk: {result['risk_score']:.1f}%")
    
    def update_charts(self, result):
        """Update visualization charts"""
        self.fig.clear()
        
        ax1 = self.fig.add_subplot(121)
        ax2 = self.fig.add_subplot(122)
        
        # Chart 1: Risk score gauge
        risk_score = result['risk_score']
        colors = ['#2ecc71' if risk_score < 30 else '#f39c12' if risk_score < 70 else '#e74c3c']
        ax1.barh(['Risk'], [risk_score], color=colors[0], height=0.5)
        ax1.set_xlim(0, 100)
        ax1.set_title('Risk Score', fontweight='bold')
        ax1.set_xlabel('Percentage (%)')
        ax1.grid(axis='x', alpha=0.3)
        
        # Chart 2: Top risk factors
        features = list(result['feature_importance'].keys())[:5]
        importance = list(result['feature_importance'].values())[:5]
        ax2.barh(features, importance, color='#3498db')
        ax2.set_title('Top Risk Factors', fontweight='bold')
        ax2.set_xlabel('Importance')
        ax2.grid(axis='x', alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def reset_form(self):
        """Reset inputs"""
        defaults = {'age': 50, 'sex': 1, 'cp': 0, 'trestbps': 120, 'chol': 200,
                   'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
                   'oldpeak': 1.0, 'slope': 1, 'ca': 0, 'thal': 2}
        for key, var in self.input_vars.items():
            var.set(defaults[key])
        self.status_var.set("🔄 Reset")
    
    def save_input(self):
        """Save input data"""
        try:
            import json
            filename = filedialog.asksaveasfilename(defaultextension=".json",
                                                   filetypes=[("JSON files", "*.json")])
            if filename:
                with open(filename, 'w') as f:
                    json.dump({k: v.get() for k, v in self.input_vars.items()}, f, indent=4)
                messagebox.showinfo("Success", "Data saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def show_help(self):
        """Show help"""
        messagebox.showinfo("Help", 
            "HEART DISEASE PREDICTION SYSTEM\n\n"
            "1. Enter patient data in the left panel\n"
            "2. Click 'Predict' to analyze\n"
            "3. View results in tabs:\n"
            "   • Summary: Quick overview\n"
            "   • Details: Full analysis\n"
            "   • Charts: Visual insights\n\n"
            "Risk Levels:\n"
            "• Low (<30%): Minimal intervention\n"
            "• Medium (30-70%): Lifestyle changes\n"
            "• High (>70%): Medical consultation")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseGUI(root)
    root.mainloop()