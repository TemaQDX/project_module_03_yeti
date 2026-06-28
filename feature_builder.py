# feature_builder.py

import numpy as np
import pandas as pd

def build_df_users_features_func(
        df_users: pd.DataFrame,
        df_visits: pd.DataFrame,
        df_ads_activity: pd.DataFrame,
        df_surf_depth: pd.DataFrame,
        df_primary_device: pd.DataFrame,
        df_cloud_usage: pd.DataFrame
    ) -> pd.DataFrame:
    """
        Строит агрегированный датасет с первичным ключом user_id на основе 6 исходных датасетов:
    df_users, df_visits, df_ads_activity, df_surf_depth, df_primary_device, df_cloud_usage
    
    df_visits трансформируется с первичным ключом user_id 
    в DataFrame с новыми признаками:
      - базовые: total_sessions, avg_sessions_per_day
      - число посещений по времени суток: count_night, count_morning, count_day, count_evening
      - число посещений по категориям сайтов: count_cat_category_X (20 шт.)
      - разнообразие: entropy_category, entropy_daytime
      - флаги: is_high_activity_user
      - отношение: ratio_evening_to_morning

    """

    # Устраняем дубликаты
    df_users = df_users.drop_duplicates()
    df_visits = df_visits.drop_duplicates()
    df_ads_activity = df_ads_activity.drop_duplicates()
    df_surf_depth = df_surf_depth.drop_duplicates()
    df_primary_device = df_primary_device.drop_duplicates()
    df_cloud_usage = df_cloud_usage.drop_duplicates()

    # Преобразование и замена значений в признаках
    df_visits['daytime'] = df_visits['daytime'].replace({'утро':'morning', 'день':'day', 'вечер':'evening','ночь':'night'})
    df_ads_activity['ads_activity'] = df_ads_activity['ads_activity'].replace({'очень часто': 'ctr_4', 'часто': 'ctr_3', 'умеренно': 'ctr_2', 'редко': 'ctr_1', 'очень редко': 'ctr_0'})
    df_surf_depth['surf_depth'] = df_surf_depth['surf_depth'].replace({'поверхностно': 'depth_0', 'средне': 'depth_1', 'глубоко': 'depth_2'})
    df_primary_device['primary_device'] = df_primary_device['primary_device'].replace({'смартфон': 'smartphone', 'ПК': 'pc', 'ноутбук': 'laptop', 'планшет': 'tablet'})
    df_cloud_usage['cloud_usage'] = df_cloud_usage['cloud_usage'].astype('int')
    
    # Трансформируем df_visits 
    df = df_visits.copy()

    # Преобразуем date в datetime
    df['date_dt'] = pd.to_datetime(df['date'])

    # -------------------------
    # Базовые агрегаты по user_id
    # -------------------------
    base_agg = (
        df.groupby('user_id')
          .agg(
              total_sessions=('session_id', 'count'),
              days_active=('date_dt', 'nunique')
          )
          .reset_index()
    )
    
    # avg_sessions_per_day: избегаем деления на 0
    base_agg['avg_sessions_per_day'] = (
        base_agg['total_sessions'] / base_agg['days_active'].replace(0, 1)
    )

    # Порог для is_high_activity_user (75 перцентиль total_sessions)
    threshold_75 = base_agg['total_sessions'].quantile(0.75)
    base_agg['is_high_activity_user'] = (base_agg['total_sessions'] > threshold_75).astype(int)

    # Убираем days_active из финального набора
    base_agg = base_agg.drop(columns=['days_active'])

    # -------------------------
    # Абсолютные значения по времени суток (daytime)
    # -------------------------
    # Считаем количество сессий для каждого daytime
    time_counts = (
        pd.crosstab(df['user_id'], df['daytime'], values=df['session_id'], aggfunc='count')
        .fillna(0)
        .astype(int)
    )
    time_counts = time_counts.add_prefix('count_')

    # ratio_evening_to_morning: отношение абсолютных значений
    # Защита от деления на ноль: заменяем 0 в знаменателе на NaN, потом заполняем 0
    ratio_evening_morning = (
        time_counts['count_evening'] / time_counts['count_morning'].replace(0, np.nan)
    ).fillna(0).rename('ratio_evening_to_morning')

    # -------------------------
    # Абсолютные значения по категориям сайтов (website_category)
    # -------------------------
    cat_counts = (
        pd.crosstab(df['user_id'], df['website_category'], values=df['session_id'], aggfunc='count')
        .fillna(0)
        .astype(int)
    )
    cat_counts = cat_counts.add_prefix('count_cat_')
    cat_counts.columns = cat_counts.columns.str.replace(' ', '_').str.lower()

    # -------------------------
    # Энтропия (разнообразие)
    # -------------------------
    
    def entropy(p: pd.Series) -> float:
        p = p[p > 0]
        if len(p) == 0:
            return 0.0
        return -np.sum(p * np.log2(p))

    # Расчет энтропии для категорий
    # Нормализуем cat_counts по строкам, чтобы получить доли для расчета энтропии
    cat_probs = cat_counts.div(cat_counts.sum(axis=1), axis=0)
    entropy_category = cat_probs.apply(entropy, axis=1).rename('entropy_category')

    # Расчет энтропии для времени суток
    # Нормализуем time_counts по строкам
    time_probs = time_counts.div(time_counts.sum(axis=1), axis=0)
    entropy_daytime = time_probs.apply(entropy, axis=1).rename('entropy_daytime')

    # -------------------------
    # Сборка итогового DataFrame
    # -------------------------
    result_visits = base_agg.merge(time_counts, on='user_id', how='left')
    result_visits = result_visits.merge(cat_counts, on='user_id', how='left')
    
    # Добавляем метрики разнообразия
    result_visits = result_visits.merge(entropy_category, left_on='user_id', right_index=True, how='left')
    result_visits = result_visits.merge(entropy_daytime, left_on='user_id', right_index=True, how='left')
    
    # Добавляем отношение
    result_visits = result_visits.merge(ratio_evening_morning, left_on='user_id', right_index=True, how='left')
    
    # Заполним NaN нулями (на случай пользователей с очень редкими категориями/временными окнами)
    result_visits = result_visits.fillna(0)

    # Финальное объединение со всеми остальными таблицами
    result_final = df_users.merge(result_visits, on='user_id', how='left')
    result_final = result_final.merge(df_ads_activity, on='user_id', how='left')
    result_final = result_final.merge(df_surf_depth, on='user_id', how='left')
    result_final = result_final.merge(df_primary_device, on='user_id', how='left')
    result_final = result_final.merge(df_cloud_usage, on='user_id', how='left')

    return result_final
