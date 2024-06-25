import numpy as np
import matplotlib.pyplot as plt

class Scheme():
    
    def __init__(self):
        self.z0 = 5
        self.x0 = 5
    
    def A(self):
        pass
    def B(self, y, n, i):
        pass
    def D(self, y, n, i):
        pass
    def F(self, y, n, i):
        pass
    
 
    def K0(self, y, n, i):
        pass
    def M0(self, y, n, i):
        pass
    def P0(self, y, n, i):
        pass
    
    def KN(self, y, n, i):
        pass
    def MN(self, y, n, i):
        pass
    def PN(self, y, n, i):
        pass
    
    def f(self, x, z):
        f0 = 15
        betha = 0.2
        return f0 * np.exp(-betha * ((x - self.x0) ** 2 + (z - self.z0) ** 2))
        # return 1

    def kappa(self,u1, u2):
        
        return (self.f(u1) + self.f(u2)) / 2

        
    def Lambda(self, u = 300):
        a1 = 0.0134 * 10
        b1 = 1
        c1 = 4.35 * 10e-4
        m1 = 1
        return a1 * (b1 + c1 * u ** m1)
    
    def rightRun(self, a, b, d, f):
        # прогоночные коэффициенты
        alpha = np.zeros(self.N)
        beta = np.zeros(self.N)
        # решение системы
        u = np.zeros(self.N)
        # прямой ход : находим a, b
        for i in range(self.N - 1):
            if i == 0:
                alpha[i + 1] = -d[i] / b[i]
                beta[i + 1] = f[i] / b[i]
            else:
                zn = alpha[i] * a[i] + b[i]
                alpha[i + 1] = -d[i] / zn
                beta[i + 1] = (f[i] - a[i] * beta[i]) / zn
                
        # обратный ход : находим u
        for i in range(self.N - 1, -1, -1):
            if i == self.N - 1:
                u[i] = (f[i] - a[i] * beta[i]) / (a[i] * alpha[i] + b[i])
                # u[i] = self.u0
            else:
                u[i] = alpha[i + 1] * u[i + 1] + beta[i + 1]
        
        return u
    
    def Solve(self, y, i):

        f = np.zeros(self.N)
        a = np.zeros(self.N)
        b = np.zeros(self.N)
        d = np.zeros(self.N)

        # A = np.zeros((self.N, self.N))

        for n in range(self.N):
            
            if n == 0:

                b[n] = self.K0(y, n, i)
                d[n] = self.M0(y, n, i)
                f[n] = self.P0(y, n, i) 
                # A[n, 0] = b[n]
                # A[n, 1] = d[n]
            elif n == self.N - 1:
                a[n] = self.KN(y, n, i)
                b[n] = self.MN(y, n, i)
                f[n] = self.PN(y, n, i)

                # A[n, n - 1] = a[n]
                # A[n, n] = b[n]
            else:
                a[n] = self.A(y, n, i) 
                b[n] = self.B(y, n, i)
                d[n] = self.D(y, n, i)
                f[n] = self.F(y, n, i)

                # A[n, n - 1] = a[n]
                # A[n, n] = b[n]
                # A[n, n + 1] = d[n]

        # print(a)
        # print(b)
        # print(d)
        # print(A)
        # print()
        # u = np.linalg.solve(A, f)
        # print(u)
        u = self.rightRun(a, b, d, f)
        return u

class xScheme(Scheme):

    def __init__(self, x, z, u0, F0, tau):
        self.N = len(x)
        self.hx = x[1] - x[0]
        self.hz = z[1] - z[0]
        self.x = x
        self.z = z
        self.u0 = u0
        self.F0 = F0
        self.tau = tau

        self.alpha2 = 1
        self.alpha3 = 1
        self.alpha4 = 1
       
        super().__init__()

    def A(self, y, n, i):
        return self.Lambda(y[i, n - 1]) / self.hx**2
    def B(self, y, n, i):
        return -(self.A(y, n, i) + self.D(y, n, i) + 2 / self.tau)
    def D(self, y, n, i):
        return self.Lambda(y[i, n]) / self.hx**2
    def F(self, y, n, i):
        
        y_before = y[i - 1, n] if i > 0 else y[0, n]
        y_after = y[i + 1, n] if i + 1 < len(y[:, n]) else y[len(y[:, n]) - 1, n]
        
        p1 = 2 * y[i, n] / self.tau
        p2 = (self.Lambda(y[i, n]) * y_after - (self.Lambda(y_before)  + self.Lambda(y[i, n])) * y[i, n] + self.Lambda(y_before) * y_before) / self.hz ** 2
        p3 = self.f(self.x[n], self.z[i]) 
        return -(p1 + p2 + p3)
    
    def K0(self, y, n, i):
        return -1
    def M0(self, y, n, i):
        return 1
    def P0(self, y, n, i):
        return  -self.F0 * self.hx / self.Lambda(y[i, n])
    
    def KN(self, y, n, i):
        return -1
    def MN(self, y, n, i):
        return 1 + self.alpha2 * self.hx / self.Lambda(y[i, n])
    def PN(self, y, n, i):
        return self.alpha2 * self.u0 * self.hx / self.Lambda(y[i, n])

class zScheme(Scheme):

    def __init__(self, x, z, u0, F0, tau):
        self.N = len(z)
        self.hz = z[1] - z[0]
        self.hx = x[1] - x[0]
        self.z = z
        self.x = x
        self.u0 = u0
        self.F0 = F0
        self.tau = tau
        
        self.alpha2 = 1
        self.alpha3 = 1
        self.alpha4 = 1
        super().__init__()

    def A(self, y, n, i):
        return self.Lambda(y[n, i]) / self.hz**2
    def B(self, y, n, i):
        return -(self.A(y, n, i) + self.D(y, n, i)  + 2 / self.tau)
    def D(self, y, n, i):
       return self.Lambda(y[n, i]) / self.hz**2
    def F(self, y, n, i):

        y_before = y[n, i - 1] if i > 0 else y[n, 0]
        y_after = y[n, i + 1] if i + 1 < len(y[n, :]) else y[n, len(y[n, :]) - 1]
        
        p1 = 2 * y[n, i] / self.tau
        p2 = (self.Lambda(y[n, i]) * y_after - (self.Lambda(y_before)  + self.Lambda(y[n, i])) * y[n, i] + self.Lambda(y_before) * y_before) / self.hx ** 2
        p3 = self.f(self.x[i], self.z[n]) 
        return -(p1 + p2 + p3)
    
    def K0(self, y, n, i):
        return (1 + self.alpha3 * self.hz/self.Lambda(y[n, i]))
    def M0(self, y, n, i):
        return -1
    def P0(self, y, n, i):
        return self.alpha3 * self.u0 * self.hz / self.Lambda(y[n, i])
    
    def KN(self, y, n, i):
        return -1
    def MN(self, y, n, i):
         return 1 + self.alpha4 * self.hz / self.Lambda(y[n, i])
    def PN(self, y, n, i):
        return self.alpha4 * self.u0 * self.hz / self.Lambda(y[n, i])

class Solver:

    def __init__(self):
        a = 10
        b = 10
        
        self.Nx = 10
        self.Nz = 10
        self.hx = a / self.Nx
        self.hz = b / self.Nz
        self.x = np.linspace(0, a, self.Nx,)
        self.z = np.linspace(0, b, self.Nz)

        self.u0 = 300
        self.F0 = 30
        self.tau = 1
        
    def GeneralSolve(self):
        x = xScheme(self.x, self.z, self.u0, self.F0, self.tau)
        z = zScheme(self.x, self.z, self.u0, self.F0, self.tau)    

        u = np.full((self.Nx, self.Nz), self.u0)
        y1 = np.zeros((self.Nx, self.Nz))
        y2 = np.zeros((self.Nx, self.Nz))
        
        flag = True
        while flag:
            
            y1 = u.copy()
            for i in range(self.Nz):
                flag2 = True
                while flag2:
                    temp = x.Solve(y1, i)
                    if np.max(np.abs((temp - y1[i, :]) / temp)) < 1e-2:
                        flag2 = False
                    y1[i, :] = temp
            
            y2 = y1.copy()
            for j in range(self.Nx):
                flag2 = True
                while flag2:
                    temp = z.Solve(y2, j)
                    if np.max(np.abs((temp - y2[:, j]) / temp)) < 1e-2:
                        flag2 = False
                    y2[:, j] = temp
                
            delta = np.max(np.abs((u - y2) / y2))
            if delta / self.tau < 2e-2:
                flag = False
            print(delta / self.tau)
            
            u = y2.copy()
            
            # x_, z_ = np.meshgrid(s.x[1:-1], s.z)
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(x_, z_, u[:, 1:-1], cmap='magma')
            # ax.set_xlabel('x')
            # ax.set_ylabel('z')
            # ax.set_zlabel('T(x, z, t)')
            # ax.set_title('Температурное поле')
            # plt.show()
            
        return u

# chem dalshe v les...
s = Solver()
u = s.GeneralSolve() / 1.7
x, z = np.meshgrid(s.x, s.z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, z, u, cmap='magma')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('u(x, z)')
ax.set_title('Температурное поле')
plt.show()

for i in range(s.Nx-1,0,-5):
    plt.plot(s.z, u[:,i], label=f'u(x,z) x={s.x[i]}', linestyle= "--" if i < 20 else "-")
plt.plot(s.z, u[:,0], label=f'u(x,z) x={s.x[0]}', linestyle= "--")
plt.grid()
plt.legend()
plt.show()
