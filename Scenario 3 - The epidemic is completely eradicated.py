import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Các tham số của mô hình
beta_0 = 0.3   # Tốc độ lây nhiễm trung bình
sigma = 1/14    # Tốc độ chuyển từ E sang I
gamma = 1/14   # Tốc độ hồi phục
u = 0.7       # Tốc độ tiêm vaccine (có thể điều chỉnh theo nhu cầu)
alpha = 0.01   # Tốc độ suy giảm miễn dịch
delta = 0.2    # Xác suất tái phát bệnh

# Thời gian và các tham số thời tiết
T = 365        # Chu kỳ của thời tiết (1 năm)
A = 0          # Biên độ dao động của beta(t)
phi = 0        # Pha ban đầu,t= pi/2 vào khoảng giữa mùa đông, t = 3pi/2 vào giữa mùa hạ.

# Hàm beta(t) theo thời gian
def beta_t(t):
    return beta_0 * (1 + A * np.sin(2 * np.pi * t / T + phi))

# Điều kiện ban đầu
S0 = 10000      # Số người dễ nhiễm
E0 = 5000       # Số người nhiễm ở trạng thái tiềm ẩn
I0 = 5000       # Số người nhiễm bệnh ở trạng thái hoạt động
R0 = 1000       # Số người hồi phục
V0 = 0          # Số người đã tiêm vaccine

# Thời gian mô phỏng (từ 0 đến 365 ngày)
t = np.linspace(0, 365, 365)

# Định nghĩa hệ phương trình vi phân
def deriv(y, t, sigma, gamma, u, alpha, delta):
    S, E, I, R, V = y
    N = S + E + I + R + V
    dSdt = -beta_t(t) * S * I / N - u * S + alpha * (1 - delta) * R
    dEdt = beta_t(t) * S * I / N - sigma * E + alpha * delta * R
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I - alpha * R
    dVdt = u * S
    return dSdt, dEdt, dIdt, dRdt, dVdt

# Điều kiện ban đầu cho hệ phương trình
y0 = S0, E0, I0, R0, V0

# Giải hệ phương trình vi phân
sol = odeint(deriv, y0, t, args=(sigma, gamma, u, alpha, delta))
S, E, I, R, V = sol.T

P_infection = (E + I) / (S + E + I + R + V)

# Vẽ đồ thị xác suất mắc bệnh theo thời gian
plt.figure(figsize=(10.5, 5))
plt.plot(t, P_infection, 'm', label='Probability of disease')
plt.xlabel('Time (Days)')
plt.ylabel('Probablity')
plt.legend(loc='best')
plt.title('Probability of disease')
plt.grid(True)
plt.show()

# Vẽ đồ thị kết quả
plt.figure(figsize=(10,5))
plt.plot(t, S, label='S(t) - Susceptible')
plt.plot(t, E, label='E(t) - Exposed')
plt.plot(t, I, label='I(t) - Infectious')
plt.plot(t, R, label='R(t) - Recovered')
plt.plot(t, V, label='V(t) - Vaccinated')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.title('SEIR Model with Vaccination, Immunity Loss, Relapse and Seasonal Variation')
plt.grid(True)
plt.show()



