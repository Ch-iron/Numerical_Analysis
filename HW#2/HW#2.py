import numpy as np
from sympy import symbols, Derivative

print("We will find the min locations fx = 5x^4 - 22.4x^3 + 15.85272x^2 + 24.161472x - 23.4824832")
print("What method do you want?")
print("Using first and second derivatives : 1, Using approximation : 2")
c = int(input("Your Choice : "))

#Using first and second derivatives
if c == 1:
    print("Your choice is Using derivatives")
    init = float(input("Input initial value : "))
    x = symbols('x')
    fx = 5*x**4 - 22.4*x**3 + 15.85272*x**2 + 24.161472*x - 23.4824832
    fd = Derivative(fx, x)
    fdev = fd.doit()
    sd = Derivative(fdev, x)
    sdev = sd.doit()
    while True:
        f1 = fdev.subs({x: init})
        f2 = sd.doit().subs({x: init})
        if f1 == 0:
            print("The min location is : ", init)
            break
        elif f1 != 0:
            nv = init - f1/f2
            if init == nv:
                print("The min location is : ", init)
                break
            elif abs(f1) < 0.000000001:
                print("The min location is : ", init)
                break
            init = nv
            
#Using approximation
elif c == 2:
    print("Your choice is Using approximation")
    xi = float(input("Input initial value : "))
    h = 0.0000001
    x = symbols('x')
    fx = 5*x**4 - 22.4*x**3 + 15.85272*x**2 + 24.161472*x - 23.4824832
    fd = Derivative(fx, x)
    fdev = fd.doit()
    sd = Derivative(fdev, x)
    sdev = sd.doit()
    while True:
        ximh = xi - h
        xiph = xi + h
        fxi = fx.subs({x: xi})
        fxiph = fx.subs({x: xiph})
        fximh = fx.subs({x: ximh})
        f1xi = (fxiph - fxi)/h
        f1ximh = (fxi - fximh)/h
        f2xi = (f1xi - f1ximh)/h
        if f1xi == 0:
            print("The min location is : ", xi)
            break
        elif f1xi != 0:
            xip1 = xi - f1xi/f2xi
            if xi == xip1:
                print("The min location is : ", xi)
                break
            elif abs(f1xi) < 0.00000001:
                print("The min location is : ", xi)
                break
            xi = xip1