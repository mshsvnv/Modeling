import numpy as np
import matplotlib.pyplot as plt

class Scheme():
    
    def __init__(self):
        self.z0 = 5
        self.x0 = 5
    
    def A(self):
        pass
    def B(self):
        pass
    def D(self):
        pass
    def F(self, y, n, i):
        pass
    
    def K0(self):
        pass
    def M0(self):
        pass
    def P0(self):
        pass
    
    def KN(self):
        pass
    def MN(self):
        pass
    def PN(self):
        pass
    
    def f(self, x, z):
        f0 = 0.5
        betha = 0.2
        return f0 * np.exp(-betha * ((x - self.x0) ** 2 + (z - self.z0) ** 2))
        # return 1
        
    def Lambda(self, u = 300):
        a1 = 0.0134
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

        for n in range(self.N):
    
            if n == 0:
                b[n] = self.K0()
                d[n] = self.M0()
                f[n] = self.P0() 
            elif n == self.N - 1:
                a[n] = self.KN()
                b[n] = self.MN()
                f[n] = self.PN()
            else:
                a[n] = self.A() 
                b[n] = self.B()
                d[n] = self.D()
                f[n] = self.F(y, n, i)

        u = self.rightRun(a, b, d, f)
        return u

class xScheme(Scheme):

    def __init__(self, x, z, u0, F0, tau):
        self.N = len(x)
        self.hx = x[1] - x[0]
        self.x = x
        self.z = z
        self.u0 = u0
        self.F0 = F0
        self.tau = tau
        super().__init__()

    def A(self):
        return 1 / self.hx**2
    def B(self):
        return -(2 / self.hx**2 + 2 / self.tau)
    def D(self):
        return 1 / self.hx**2
    def F(self, y, n, i):

        if i == 0:
            p1 = 2 * y[i, n] / self.tau
            p2 = (y[0, n] - 2 * y[1, n] + y[2, n]) / self.hx ** 2
            p3 = self.f(self.x[n], self.z[i]) / self.Lambda()
        elif i == len(y[:, n]) - 1:
            p1 = 2 * y[i, n] / self.tau
            p2 = (y[i - 2, n] - 2 * y[i - 1, n] + y[i, n]) / self.hx ** 2
            p3 = self.f(self.x[n], self.z[i]) / self.Lambda()
        else:
            p1 = 2 * y[i, n] / self.tau
            p2 = (y[i - 1, n] - 2 * y[i, n] + y[i + 1, n]) / self.hx ** 2
            p3 = self.f(self.x[n], self.z[i]) / self.Lambda()

        
        # y_before = y[i - 1, n] if i > 0 else y[0, n]
        # y_after = y[i + 1, n] if i + 1 < len(y[:, n]) else y[len(y[:, n]) - 1, n]
        
        # p1 = 2 * y[i, n] / self.tau
        # p2 = (y_before - 2 * y[i, n] + y_after) / self.hx ** 2
        # p3 = self.f(self.x[n], self.z[i]) / self.Lambda()
        return -(p1 + p2 + p3)
    
    def K0(self):
        return 1
    def M0(self):
        return 0
    def P0(self):
        return self.u0
    
    def KN(self):
        return 1
    def MN(self):
        return 0
    def PN(self):
        return self.u0

class zScheme(Scheme):

    def __init__(self, x, z, u0, F0, tau):
        self.N = len(z)
        self.hz = z[1] - z[0]
        self.z = z
        self.x = x
        self.u0 = u0
        self.F0 = F0
        self.tau = tau
        super().__init__()

    def A(self):
        return 1 / self.hz**2
    def B(self):
        return -(2 / self.hz**2 + 2 / self.tau)
    def D(self):
        return 1 / self.hz**2
    def F(self, y, n, i):

        y_before = y[n, i - 1] if i > 0 else y[n, 0]
        y_after = y[n, i + 1] if i + 1 < len(y[n, :]) else y[n, len(y[n, :]) - 1]
        
        p1 = 2 * y[n, i] / self.tau
        p2 = (y_before - 2 * y[n, i] + y_after) / self.hz ** 2
        p3 = self.f(self.x[i], self.z[n]) / self.Lambda()
        return -(p1 + p2 + p3)
    
    def K0(self):
        return 1
    def M0(self):
        return 0
    def P0(self):
        return self.u0
    
    def KN(self):
        return 1
    def MN(self):
        return 0
    def PN(self):
        return self.u0

class Solver:

    def __init__(self, Nx, Nz, EPS):
        a = 10
        b = 10
        
        self.EPS = EPS
        self.Nx = Nx
        self.Nz = Nz
        self.hx = a / self.Nx
        self.hz = b / self.Nz
        self.x = np.linspace(0, a, self.Nx,)
        self.z = np.linspace(0, b, self.Nz)

        self.u0 = 300
        self.F0 = 30
        self.tau = 20
        
    def GeneralSolve(self):
        x = xScheme(self.x, self.z, self.u0, self.F0, self.tau)
        z = zScheme(self.x, self.z, self.u0, self.F0, self.tau)    

        u = np.full((self.Nx, self.Nz), self.u0)
        y1 = np.zeros((self.Nx, self.Nz))
        y2 = np.zeros((self.Nx, self.Nz))
        
        flag = True
        while flag:
            
            for i in range(self.Nz):
                y1[i,:] = x.Solve(u, i)
            
            for j in range(self.Nx):
                y2[:,j] = z.Solve(y1, j)
                
            delta = np.max(np.abs((u - y2) / y2))
            if delta / self.tau < self.EPS:
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
s = Solver(110, 110, 1e-3)
u = s.GeneralSolve()
x, z = np.meshgrid(s.x[1:-1], s.z)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, z, u[:, 1:-1], cmap='magma')
ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('u(x, z)')
ax.set_title('Температурное поле')
plt.show()

u = s.GeneralSolve()
for i in range(0,s.Nx,11):
    plt.plot(s.z[:-1], u[:-1,i+1], label=f'u(x,z) x={s.x[i+1]}', linestyle= "--" if i > 55 else "-")
plt.plot(s.z[:-1], u[:-1,-2], label=f'u(x,z) x={s.x[-2]}', linestyle= "--")
plt.grid()
plt.legend()
plt.show()
