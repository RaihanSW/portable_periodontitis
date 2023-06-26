def InputDetection(option):
    if option == "quit":
        return "quit"
    

    try:
        option = int(option)
        if option < 1 or option > 5:
            raise ValueError
        else:
            print(f"the choice is number : {option}")
            return option
    except ValueError:
        print("Option is not available")
    except:
        print("Must input number")
    