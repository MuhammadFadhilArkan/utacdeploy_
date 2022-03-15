from pydantic import BaseModel, conlist, ValidationError

class Ct3hours(BaseModel):
    mean_MatteTIn_Curent3_Amp:float
    mean_Blower_motor_R_Current_Amp:float
    mean_Blower_motor_S_Current_Amp:float
    mean_Blower_motor_T_Current_Amp:float
    std_Converyer_Belt_Speed_m_min:float
    std_Blower_Pressure_Bar:float
    std_MatteTIn_Curent3_Amp:float
    std_Blower_motor_R_Current_Amp:float
    std_Blower_motor_S_Current_Amp:float
    std_Blower_motor_T_Current_Amp:float
    min_Converyer_Belt_Speed_m_min:float
    min_Blower_motor_R_Current_Amp:float
    min_Blower_motor_S_Current_Amp:float
    min_Blower_motor_T_Current_Amp:float
    max_MatteTIn_Curent5_Amp:float
    max_Blower_motor_R_Current_Amp:float
    qntl1_MatteTIn_Curent2_Amp:float
    qntl1_MatteTIn_Curent3_Amp:float
    qntl1_Blower_motor_R_Current_Amp:float
    qntl1_Blower_motor_S_Current_Amp:float
    qntl1_Blower_motor_T_Current_Amp:float
    qntl3_MatteTIn_Curent2_Amp:float
    qntl3_MatteTIn_Curent3_Amp:float
    qntl3_Blower_motor_S_Current_Amp:float
    median_MatteTIn_Curent3_Amp:float
    median_Blower_motor_R_Current_Amp:float
    median_Blower_motor_S_Current_Amp:float
    median_Blower_motor_T_Current_Amp:float
    dow:float