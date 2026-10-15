import numpy as np
import pandas as pd

# --- Параметры для платинового ТС (α = 0.00385 °C⁻¹) по ГОСТ 6651-2009, п. 5.2.1 ---
R0 = 100.0  # Номинальное сопротивление при 0 °C, Ом
A = 3.9083e-3
B = -5.775e-7
C = -4.183e-12  # Для диапазона от -200 до 0 °C

def resistance_from_temperature(t):
    """
    Прямая функция: сопротивление по температуре.
    t: температура в °C (скаляр или numpy-массив)
    """
    t = np.asarray(t, dtype=float)
    R = np.zeros_like(t)

    # Диапазон от -200 до 0 °C (формула 1)
    mask_neg = t < 0
    R[mask_neg] = R0 * (1 + A * t[mask_neg] + B * t[mask_neg]**2 +
                         C * (t[mask_neg] - 100) * t[mask_neg]**3)

    # Диапазон от 0 до 850 °C (формула 2)
    mask_pos = t >= 0
    R[mask_pos] = R0 * (1 + A * t[mask_pos] + B * t[mask_pos]**2)

    return R

def temperature_from_resistance(R):
    """
    Обратная функция: температура по сопротивлению.
    R: сопротивление в Ом (скаляр или numpy-массив)
    """
    R = np.asarray(R, dtype=float)
    t = np.zeros_like(R)

    # Для диапазона R >= R0 (t >= 0 °C) — точное аналитическое решение
    mask_pos = R >= R0
    # Решаем квадратное уравнение: R0*(1 + A*t + B*t^2) = R
    # B*t^2 + A*t + (1 - R/R0) = 0
    disc = A**2 - 4 * B * (1 - R[mask_pos] / R0)
    t[mask_pos] = (-A + np.sqrt(disc)) / (2 * B)

    # Для диапазона R < R0 (t < 0 °C) — итерационное решение (метод Ньютона)
    mask_neg = R < R0
    if np.any(mask_neg):
        R_neg = R[mask_neg]
        # Начальное приближение
        t_neg = (R_neg / R0 - 1) / A
        for _ in range(50):
            # Функция и её производная
            f = R0 * (1 + A * t_neg + B * t_neg**2 +
                      C * (t_neg - 100) * t_neg**3) - R_neg
            df = R0 * (A + 2 * B * t_neg +
                       C * (4 * t_neg**3 - 300 * t_neg**2))
            t_neg = t_neg - f / df
        t[mask_neg] = t_neg

    return t

# --- Генерация датасета ---
# Диапазон температур по ГОСТ 6651-2009 для платинового ТС: от -200 до +850 °C
# Шаг изменения температуры: 1.5 °C
step = 1.5
temperatures = np.arange(-200, 850 + step, step)

# 1. Моделирование сопротивления по температуре
resistances = resistance_from_temperature(temperatures)

# 2. Обратное моделирование: температура по сопротивлению (без погрешностей)
temperatures_recovered = temperature_from_resistance(resistances)

# --- Формирование датасета ---
df = pd.DataFrame({
    'temperature_C': temperatures,
    'resistance_Ohm': resistances,
    'temperature_recovered_C': temperatures_recovered,
    'temperature_error_C': temperatures_recovered - temperatures
})

# Сохранение в CSV
df.to_csv('pt100_dataset.csv', index=False)

print("Датасет сгенерирован и сохранён в 'pt100_dataset.csv'")
print(f"Количество точек: {len(df)}")
print(f"Диапазон температур: от {temperatures[0]:.1f} до {temperatures[-1]:.1f} °C")
print(f"Шаг изменения температуры: {step} °C")
print(f"Максимальная ошибка обратного преобразования: {np.max(np.abs(df['temperature_error_C'])):.2e} °C")
print("\nПервые 10 строк:")
print(df.head(10).to_string(index=False))
print("\nПоследние 5 строк:")
print(df.tail(5).to_string(index=False))