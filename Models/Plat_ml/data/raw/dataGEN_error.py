import numpy as np
import pandas as pd

# --- Параметры для платинового ТС (α = 0.00385 °C⁻¹) по ГОСТ 6651-2009, п. 5.2.1 ---
R0 = 100.0  # Номинальное сопротивление при 0 °C, Ом
A = 3.9083e-3
B = -5.775e-7
C = -4.183e-12  # Для диапазона от -200 до 0 °C

def resistance_from_temperature(t):
    """Прямая функция: сопротивление по температуре (НСХ)."""
    t = np.asarray(t, dtype=float)
    R = np.zeros_like(t)
    mask_neg = t < 0
    R[mask_neg] = R0 * (1 + A * t[mask_neg] + B * t[mask_neg]**2 +
                         C * (t[mask_neg] - 100) * t[mask_neg]**3)
    mask_pos = t >= 0
    R[mask_pos] = R0 * (1 + A * t[mask_pos] + B * t[mask_pos]**2)
    return R

def temperature_from_resistance(R):
    """Обратная функция: температура по сопротивлению."""
    R = np.asarray(R, dtype=float)
    t = np.zeros_like(R)
    mask_pos = R >= R0
    disc = A**2 - 4 * B * (1 - R[mask_pos] / R0)
    t[mask_pos] = (-A + np.sqrt(disc)) / (2 * B)
    mask_neg = R < R0
    if np.any(mask_neg):
        R_neg = R[mask_neg]
        t_neg = (R_neg / R0 - 1) / A
        for _ in range(50):
            f = R0 * (1 + A * t_neg + B * t_neg**2 +
                      C * (t_neg - 100) * t_neg**3) - R_neg
            df = R0 * (A + 2 * B * t_neg +
                       C * (4 * t_neg**3 - 300 * t_neg**2))
            t_neg = t_neg - f / df
        t[mask_neg] = t_neg
    return t

def tolerance_AA(t):
    """Допуск класса AA в °C по ГОСТ 6651-2009."""
    return 0.1 + 0.0017 * np.abs(t)

def tolerance_C(t):
    """Допуск класса C в °C по ГОСТ 6651-2009."""
    return 0.6 + 0.01 * np.abs(t)

# --- Генерация датасета ---
step = 1.5
temperatures = np.arange(-200, 850 + step, step)
N = len(temperatures)

# Номинальное сопротивление по НСХ
R_nominal = resistance_from_temperature(temperatures)

# Чувствительность dR/dt (Ом/°C)
dR_dt = np.gradient(R_nominal, temperatures)

# Маска классов: 70% AA, 30% C
np.random.seed(42)
class_mask = np.random.rand(N) < 0.7  # True = AA, False = C

# Случайная погрешность в пределах допуска соответствующего класса
tolerance_C_values = np.where(class_mask, tolerance_AA(temperatures), tolerance_C(temperatures))
delta_t = np.random.uniform(-1, 1, N) * tolerance_C_values
delta_R = delta_t * dR_dt

# Измеренное сопротивление = номинальное + шум
R_measured = R_nominal + delta_R

# Восстановленная температура
t_recovered = temperature_from_resistance(R_measured)

# --- Формирование датасета (только нужные столбцы) ---
df = pd.DataFrame({
    'temperature_C': temperatures,
    'resistance_Ohm': R_measured,
    'temperature_recovered_C': t_recovered,
    'temperature_error_C': t_recovered - temperatures
})

df.to_csv('pt100_dataset_AA70_C30.csv', index=False)

# --- Статистика ---
err = df['temperature_error_C'].values
tol = tolerance_C_values
within = np.abs(err) <= tol
print("Датасет сохранён в 'pt100_dataset_AA70_C30.csv'")
print(f"Всего точек: {len(df)}")
print(f"Класс AA: {class_mask.sum()} точек ({class_mask.mean()*100:.1f}%)")
print(f"Класс C:  {(~class_mask).sum()} точек ({(~class_mask).mean()*100:.1f}%)")
print(f"Макс. |ошибка|: {np.max(np.abs(err)):.4f} °C")
print(f"СКО ошибки:    {np.std(err):.4f} °C")
print(f"Точек в пределах допуска: {within.sum()} / {len(df)} ({within.mean()*100:.1f}%)")
print("\nПервые 10 строк:")
print(df.head(10).to_string(index=False))