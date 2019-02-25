var = 99

def local():
    var = 0  # Change local var
def glob1():
    global var  # Declare global (normal) # Change global var
    var += 1
def glob2():
    var = 0  # Change local var
    import thismodule     # Import myself
    thismodule.var += 1  # Change global var
def glob3():
    var = 0  # Change local var
    import sys  # Import system table
    glob = sys.modules['thismodule']  # Get module object (or use __name__)
    glob.var += 1  # Change global var
def test():
    print(var)
    local()
    glob1()
    glob2()
    glob3()
    print(var)