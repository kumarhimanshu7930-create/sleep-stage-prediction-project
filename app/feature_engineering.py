import numpy as np


def create_features(heart, motion):

    heart = heart.copy()
    motion = motion.copy()

    # Rename motion columns to match training feature names
    motion = motion.rename(
        columns={
            "acceleration_x_g": "acc_x",
            "acceleration_y_g": "acc_y",
            "acceleration_z_g": "acc_z",
        }
    )

    # 1. CREATE 30-SECOND EPOCHS

    heart["epoch"] = (heart["seconds"] // 30) * 30
    motion["epoch"] = (motion["seconds"] // 30) * 30

    # 2. HEART RATE FEATURES

    heart_features = (
        heart.groupby("epoch")["heart_rate_bpm"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "heart_rate_bpm_mean",
                "min": "heart_rate_bpm_min",
                "max": "heart_rate_bpm_max",
                "std": "heart_rate_bpm_std",
            }
        )
    )

    # 3. ACCELERATION MAGNITUDE

    motion["acc_magnitude"] = np.sqrt(
        motion["acc_x"] ** 2 + motion["acc_y"] ** 2 + motion["acc_z"] ** 2
    )

    # 4. MOTION FEATURES

    motion_features = motion.groupby("epoch").agg(
        {
            "acc_x": ["mean", "std"],
            "acc_y": ["mean", "std"],
            "acc_z": ["mean", "std"],
            "acc_magnitude": ["mean", "std"],
        }
    )

    motion_features.columns = [
        "_".join(str(x) for x in col if x) for col in motion_features.columns
    ]

    motion_features = motion_features.reset_index()

    # 5. MERGE SENSOR FEATURES

    df = heart_features.merge(motion_features, on="epoch", how="outer")

    df = df.sort_values("epoch").reset_index(drop=True)

    # 6. LAG FEATURES

    df["motion_lag1"] = df["acc_magnitude_std"].shift(1)

    # 7. HEART RATE ROLLING FEATURES

    df["hr_rolling_std_5min"] = (
        df["heart_rate_bpm_mean"].rolling(10, min_periods=2).std()
    )

    df["hr_range_5min"] = (
        df["heart_rate_bpm_mean"].rolling(10, min_periods=2).max()
        - df["heart_rate_bpm_mean"].rolling(10, min_periods=2).min()
    )

    df["hr_range"] = df["heart_rate_bpm_max"] - df["heart_rate_bpm_min"]

    df["hr_rolling_mean_2min"] = (
        df["heart_rate_bpm_mean"].rolling(4, min_periods=2).mean()
    )

    df["hr_rolling_std_2min"] = (
        df["heart_rate_bpm_mean"].rolling(4, min_periods=2).std()
    )

    # 8. ACCELERATION ROLLING FEATURES

    df["acc_rolling_std_5min"] = (
        df["acc_magnitude_std"].rolling(10, min_periods=2).mean()
    )

    df["acc_rolling_mean_5min"] = (
        df["acc_magnitude_mean"].rolling(10, min_periods=2).mean()
    )

    df["acc_rolling_std_2min"] = (
        df["acc_magnitude_std"].rolling(4, min_periods=2).mean()
    )

    df["acc_rolling_mean_2min"] = (
        df["acc_magnitude_mean"].rolling(4, min_periods=2).mean()
    )

    # 9. ADDITIONAL FEATURES

    df["hr_change_abs"] = df["heart_rate_bpm_mean"].diff().abs()

    df["motion_change_abs"] = df["acc_magnitude_std"].diff().abs()

    return df
