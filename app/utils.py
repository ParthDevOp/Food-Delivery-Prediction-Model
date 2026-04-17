import pandas as pd

def prepare_input(dist, prep, exp, tod, veh, traf, weat):
    traffic_map = {'Low': 1, 'Medium': 2, 'High': 3}

    data = {
        'Distance_km': dist,
        'Preparation_Time_min': prep,
        'Courier_Experience_yrs': exp,
        'Time_of_Day': tod,
        'Vehicle_Type': veh,
        'Traffic_Level': traffic_map.get(traf, 2),
        'Weather': weat
    }

    return pd.DataFrame([data])