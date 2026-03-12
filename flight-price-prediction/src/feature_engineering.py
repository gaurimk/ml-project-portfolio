import pandas as pd

class FeatureEngineering:

    def transform(self, df):

        # Journey Date
        df["Journey_day"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.day
        df["Journey_month"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y").dt.month
        df.drop(["Date_of_Journey"], axis=1, inplace=True)

        # Departure Time
        df["Dep_hour"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.hour
        df["Dep_min"] = pd.to_datetime(df["Dep_Time"], format="%H:%M").dt.minute
        df.drop(["Dep_Time"], axis=1, inplace=True)

        # Arrival Time
        df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
        df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
        df.drop(["Arrival_Time"], axis=1, inplace=True)

        # Duration
        duration = list(df["Duration"])

        for i in range(len(duration)):
            if len(duration[i].split()) != 2:
                if "h" in duration[i]:
                    duration[i] = duration[i] + " 0m"
                else:
                    duration[i] = "0h " + duration[i]

        df["Duration"] = duration

        df["Duration_hours"] = df["Duration"].apply(lambda x: int(x.split()[0][:-1]))
        df["Duration_mins"] = df["Duration"].apply(lambda x: int(x.split()[1][:-1]))

        df.drop(["Duration"], axis=1, inplace=True)

        # Total Stops
        df.replace({
            "Total_Stops":{
                "non-stop":0,
                "1 stop":1,
                "2 stops":2,
                "3 stops":3,
                "4 stops":4
            }
        }, inplace=True)

        # One Hot Encoding
        airline = pd.get_dummies(df["Airline"], drop_first=True)
        source = pd.get_dummies(df["Source"], drop_first=True)
        destination = pd.get_dummies(df["Destination"], drop_first=True)

        df.drop(["Airline","Source","Destination","Route","Additional_Info"], axis=1, inplace=True)

        df = pd.concat([df, airline, source, destination], axis=1)

        return df