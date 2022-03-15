from pydantic import BaseModel, conlist, ValidationError

class St168hours(BaseModel):
    mean_Converyer_Belt_Speed_m_min:float
    mean_Blower_Pressure_Bar:float
    mean_Temp_test_degree_C:float
    mean_MatteTIn_Curent1_Amp:float
    mean_MatteTIn_Curent5_Amp:float
    mean_Blower_motor_R_Current_Amp:float
    std_Blower_Pressure_Bar:float
    std_Temp_test_degree_C:float
    std_MatteTIn_Curent1_Amp:float
    std_MatteTIn_Curent2_Amp:float
    std_MatteTIn_Curent5_Amp:float
    std_Blower_motor_R_Current_Amp:float
    std_Blower_motor_S_Current_Amp:float
    std_Blower_motor_T_Current_Amp:float
    min_Converyer_Belt_Speed_m_min:float
    min_Blower_Pressure_Bar:float
    min_Blower_motor_R_Current_Amp:float
    min_Blower_motor_T_Current_Amp:float
    max_Converyer_Belt_Speed_m_min:float
    max_Blower_Pressure_Bar:float
    max_Temp_test_degree_C:float
    max_MatteTIn_Curent1_Amp:float
    max_MatteTIn_Curent2_Amp:float
    max_MatteTIn_Curent4_Amp:float
    max_MatteTIn_Curent5_Amp:float
    max_Blower_motor_R_Current_Amp:float
    qntl1_Blower_Pressure_Bar:float
    qntl1_MatteTIn_Curent3_Amp:float
    qntl3_Blower_Pressure_Bar:float
    qntl3_Blower_motor_R_Current_Amp:float
    qntl3_Blower_motor_S_Current_Amp:float
    median_Blower_Pressure_Bar:float
    median_MatteTIn_Curent3_Amp:float
    median_Blower_motor_R_Current_Amp:float
    rows:float
    dow:float
    hod:float
    type:float