from pydantic import BaseModel, conlist, ValidationError

class AlarmBinary(BaseModel):
    mean_Converyer_Belt_Speed_m_min:float
    median_Converyer_Belt_Speed_m_min:float
    std_Converyer_Belt_Speed_m_min:float
    qntl1_Converyer_Belt_Speed_m_min:float
    qntl3_Converyer_Belt_Speed_m_min:float
    min_Converyer_Belt_Speed_m_min:float
    max_Converyer_Belt_Speed_m_min:float
    mean_Blower_Pressure_Bar:float
    median_Blower_Pressure_Bar:float
    std_Blower_Pressure_Bar:float
    qntl1_Blower_Pressure_Bar:float
    qntl3_Blower_Pressure_Bar:float
    min_Blower_Pressure_Bar:float
    max_Blower_Pressure_Bar:float
    mean_MatteTIn_Curent_Amp_Avg:float
    median_MatteTIn_Curent_Amp_Avg:float
    std_MatteTIn_Curent_Amp_Avg:float
    qntl1_MatteTIn_Curent_Amp_Avg:float
    qntl3_MatteTIn_Curent_Amp_Avg:float
    min_MatteTIn_Curent_Amp_Avg:float
    max_MatteTIn_Curent_Amp_Avg:float       