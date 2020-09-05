import numpy as np
from sympy import symbols, Derivative

print("We will solve the equation 5x^4 - 22.4x^3 + 15.85272x^2 + 24.161472x - 23.4824832")
print("What method do you want?")
print("Bisection : 1, Newton-Raphson : 2")
c = int(input("Your Choice : "))

#Bisection
if c == 1:
    print("Your choice is Bisection")
    def fx(x):
        result = 5*x**4 - 22.4*x**3 + 15.85272*x**2 + 24.161472*x - 23.4824832
        return result
    min = float(input("Input interval's Min value : "))
    max = float(input("Input interval's Max value : "))
    sc = fx(min) * fx(max)
    while True:
        if min >= max:
            print("Your input is wrong interval")
            min = float(input("Input interval's Min value : "))
            max = float(input("Input interval's Max value : "))
            sc = fx(min) * fx(max)
            continue

        if sc == 0:
            if fx(min) == 0:
                print("The root is", min)
                break
            elif fx(max) == 0:
                print("The root is", max)
                break 
        
        elif sc > 0:
            print("Your input is wrong interval")
            min = float(input("Input interval's Min value : "))
            max = float(input("Input interval's Max value : "))
            sc = fx(min) * fx(max)
            continue
        
        elif sc < 0:
            if max - min < 0.0001:
                print("The root exists between %f, %f" % (min, max))
                break
            while True:
                max2 = max - (max - min)/2
                sc = fx(min) * fx(max2)
                if sc > 0:
                    min = max2
                    sc = fx(min) * fx(max)
                    continue
                
                elif sc < 0:
                    if max - min < 0.0001:
                        print("The root exists between %f, %0.12f" % (min, max))
                        break
                    max = max2
                    continue

                elif sc == 0:
                    if fx(min) == 0:
                        print("The root is", min)
                        break
                    elif fx(max2) == 0:
                        print("The root is", max2)
                        break
            break        

#Newton-Raphson
elif c == 2:
    print("Your choice is Newton-Raphson")
    init = float(input("Input initial value : "))
    x = symbols('x')
    fx = 5*x**4 - 22.4*x**3 + 15.85272*x**2 + 24.161472*x - 23.4824832
    while True:
        v = fx.subs({x: init})
        if v == 0:
            print("The root is", init)
            break
        elif v != 0:
            lim = Derivative(fx, x)
            d = lim.doit().subs({x: init})
            nv = init - v/d
            if abs((nv - init)/nv) * 100 < 0.0001:
                print("The approximated root is", nv)
                break
            init = nv