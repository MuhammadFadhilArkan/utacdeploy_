from pydantic import BaseModel, conlist, ValidationError

class Ct168hours(BaseModel):
    mean_Converyer_Belt_Speed_m_min:float
    mean_MatteTIn_Curent1_Amp:float
    mean_MatteTIn_Curent3_Amp:float
    mean_MatteTIn_Curent5_Amp:float
    mean_Blower_motor_R_Current_Amp:float
    mean_Blower_motor_T_Current_Amp:float
    std_Converyer_Belt_Speed_m_min:float
    std_MatteTIn_Curent1_Amp:float
    std_MatteTIn_Curent4_Amp:float
    std_MatteTIn_Curent5_Amp:float
    std_Blower_motor_T_Current_Amp:float
    min_Converyer_Belt_Speed_m_min:float
    max_Converyer_Belt_Speed_m_min:float
    max_MatteTIn_Curent1_Amp:float
    max_MatteTIn_Curent4_Amp:float
    max_MatteTIn_Curent5_Amp:float
    max_Blower_motor_R_Current_Amp:float
    max_Blower_motor_T_Current_Amp:float
    qntl1_Converyer_Belt_Speed_m_min:float
    qntl1_Blower_motor_R_Current_Amp:float
    qntl1_Blower_motor_T_Current_Amp:float
    qntl3_Converyer_Belt_Speed_m_min:float
    qntl3_Blower_motor_R_Current_Amp:float
    qntl3_Blower_motor_T_Current_Amp:float
    median_Converyer_Belt_Speed_m_min:float
    median_Blower_motor_R_Current_Amp:float
    median_Blower_motor_T_Current_Amp:float
    rows:float
    dow:float