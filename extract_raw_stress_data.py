import csv
import json

USER_ID = 10
CLASSIFICATION = "S2"   
FILEPATH = "../../Machine_Learning_Data/Stroop_Trial_2/stroop_trial_2_as.json"

def extract_data(json_data):
    users_data = json_data.get("UsersData", {})
    extracted_data = []
    
    for user_id, user_data in users_data.items():
        mask_data = user_data.get("mask_data", {})
        shirt_data = user_data.get("shirt_data", {})
        
        # Get all unique timestamps from both mask_data and shirt_data
        all_timestamps = sorted(set(mask_data.keys()).union(set(shirt_data.keys())))
        
        # Track the most recent values for mask and shirt data
        last_mask_data = {}
        last_shirt_data = {}
        
        for timestamp in all_timestamps:
            row = {
                "classification": CLASSIFICATION,
                "user_id": USER_ID,
                "timestamp": timestamp,
                "CO2": last_mask_data.get("CO2", ""),
                "VOC": last_mask_data.get("VOC", ""),
                "abdomen_coil": last_shirt_data.get("abdomen_coil", ""),
                "chest_coil": last_shirt_data.get("chest_coil", ""),
                "gsr": last_shirt_data.get("gsr", ""),
                "ppg_ir": last_shirt_data.get("ppg_ir", ""),
                "ppg_red": last_shirt_data.get("ppg_red", "")
            }
            
            # Update the row with current mask_data if available
            if timestamp in mask_data:
                last_mask_data = mask_data[timestamp]
                row["CO2"] = last_mask_data.get("CO2", row["CO2"])
                row["VOC"] = last_mask_data.get("VOC", row["VOC"])
            
            # Update the row with current shirt_data if available
            if timestamp in shirt_data:
                last_shirt_data = shirt_data[timestamp]
                row["abdomen_coil"] = last_shirt_data.get("abdomen_coil", row["abdomen_coil"])
                row["chest_coil"] = last_shirt_data.get("chest_coil", row["chest_coil"])
                row["gsr"] = last_shirt_data.get("gsr", row["gsr"])
                row["ppg_ir"] = last_shirt_data.get("ppg_ir", row["ppg_ir"])
                row["ppg_red"] = last_shirt_data.get("ppg_red", row["ppg_red"])
            
            extracted_data.append(row)
    
    return extracted_data

def write_to_csv(data, filename):
    fieldnames = ["classification", "user_id", "timestamp", "CO2", "VOC", "abdomen_coil", "chest_coil", "gsr", "ppg_ir", "ppg_red"]
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

if __name__ == "__main__":
    with open(FILEPATH, "r") as f:
        json_data = json.load(f)
    
    extracted_data = extract_data(json_data)
    write_to_csv(extracted_data, "../../Machine_Learning_Data/stress_raw_data.csv")
    print("Data extracted and written to stress_raw_data.csv")
