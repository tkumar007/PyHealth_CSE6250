import os
from datetime import datetime
from typing import Any
import torch
from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import BaseTask

def generate_tpc_tensors(sample: dict) -> dict:
    """
    Dynamically generates the vitals_mask and vitals_time_delta 
    tensors expected by the Temporal Pointwise Convolution model.
    """
    vitals = sample["vitals"]
    seq_len, num_features = vitals.shape
    
    # Binary Mask (1 if measured, 0 if missing/padded)
    vitals_mask = (vitals != 0).float()
    
    vitals_time_delta = torch.zeros_like(vitals)
    time_since_measured = torch.zeros(num_features)
    
    for t in range(seq_len):
        time_since_measured += 1.0 
        
        measured_indices = (vitals_mask[t] == 1.0)
        time_since_measured[measured_indices] = 0.0
        
        vitals_time_delta[t] = time_since_measured.clone()
        
    sample["vitals_mask"] = vitals_mask
    sample["vitals_time_delta"] = vitals_time_delta
    
    return sample

class LengthOfStayTask(BaseTask):
    """
    Extracts Length of Stay in days as the regression target.
    """
    task_name = "Length of Stay Prediction"
    
    input_schema = {
        "static_demographics": "sequence",
        "vitals": "tensor" 
    }
    
    output_schema = {
        "los": "regression"
    }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        samples = []
        admissions = patient.get_events(event_type="admissions")
        
        for admission in admissions:
            admit_time = admission.timestamp
            dischtime_str = getattr(admission, 'dischtime', None)
            
            if admit_time is None or dischtime_str is None:
                continue
                
            discharge_time = datetime.strptime(dischtime_str, "%Y-%m-%d %H:%M:%S")
            los_days = (discharge_time - admit_time).total_seconds() / 86400.0
            
            if los_days <= 0:
                continue
                
            patients_info = patient.get_events(event_type="patients")
            gender = patients_info[0].gender if patients_info and hasattr(patients_info[0], 'gender') else "UNKNOWN"
            
            # Placeholder structured tensor for the admission window
            vitals_tensor = torch.zeros((24, 12), dtype=torch.float32)
            
            sample = {
                "visit_id": admission.hadm_id,
                "patient_id": patient.patient_id,
                "static_demographics": [gender], 
                "vitals": vitals_tensor, 
                "los": float(los_days),
            }
            samples.append(sample)
            
        return samples


if __name__ == "__main__":
    mimic4_root = "./data/mimiciv/demo" 
    
    if not os.path.exists(mimic4_root):
        print(f"Please download the MIMIC-IV demo data and extract it to {mimic4_root}")
    else:
        print("Initializing MIMIC4Dataset...")
        dataset = MIMIC4Dataset(
            ehr_root=mimic4_root,
            ehr_tables=["patients", "admissions", "labevents"]
        ) 
        
        print("Applying Length of Stay Task...")
        sample_dataset = dataset.set_task(LengthOfStayTask())
        
        print(f"Total samples created: {len(sample_dataset)}")
        
        print(f"Input Schema: {sample_dataset.input_schema}")
        print(f"Output Schema: {sample_dataset.output_schema}")
        
        if len(sample_dataset) > 0:
            print("Sample 0:", sample_dataset[0])
        else:
            print("No valid samples were generated.")