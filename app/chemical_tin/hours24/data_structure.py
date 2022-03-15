from pydantic import BaseModel, conlist, ValidationError

class Ct24hours(BaseModel):
    mean_Blower_Pressure_Bar:float
    mean_Blower_motor_R_Current_Amp:float
    mean_Blower_motor_S_Current_Amp:float
    std_Converyer_Belt_Speed_m_min:float
    std_Temp_test_degree_C:float
    std_MatteTIn_Curent1_Amp:float
    std_Blower_motor_R_Current_Amp:float
    std_Blower_motor_S_Current_Amp:float
    std_Blower_motor_T_Current_Amp:float
    min_Converyer_Belt_Speed_m_min:float
    min_Blower_motor_R_Current_Amp:float
    min_Blower_motor_S_Current_Amp:float
    min_Blower_motor_T_Current_Amp:float
    max_Converyer_Belt_Speed_m_min:float
    max_Blower_Pressure_Bar:float
    max_Temp_test_degree_C:float
    max_Blower_motor_S_Current_Amp:float
    qntl1_Converyer_Belt_Speed_m_min:float
    qntl1_Blower_Pressure_Bar:float
    qntl1_Blower_motor_R_Current_Amp:float
    qntl1_Blower_motor_S_Current_Amp:float
    qntl3_Converyer_Belt_Speed_m_min:float
    qntl3_Blower_Pressure_Bar:float
    qntl3_Blower_motor_R_Current_Amp:float
    qntl3_Blower_motor_S_Current_Amp:float
    median_Converyer_Belt_Speed_m_min:float
    median_Blower_motor_S_Current_Amp:float
    rows:float
    dow:float