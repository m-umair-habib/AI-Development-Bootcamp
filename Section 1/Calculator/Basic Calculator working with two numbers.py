# -----Developed By: Muhammad Umair Habib-----

# function to add two numbers
def add(x, y):
    return x + y

# function to subtract two numbers
def sub(x, y):
    return x - y

# function to multiply two numbers
def mul(x, y):
    return x * y

# function to divide two numbers
def div(x, y):
    return x / y


def calculator():
    print("Select operation: ")
    print("1- Add")
    print("2- Subtract")
    print("3- Multiply")
    print("4- Divide")

    while True:
        # Take operation's input from user
        choice = input("Enter a choice: ")

        # Check if the input is one of the four options
        if choice in ['1', '2', '3', '4']:
            num1 = float(input("Enter the first number: "))
            num2 = float(input("Enter the second number: "))

            if choice == '1':
                print(add(num1, num2))
            
            if choice == '2':
                print(sub(num1, num2))

            if choice == '3':
                print(mul(num1, num2))

            if choice == '4':
                print(div(num1, num2))

        # option to exit the while loop
        next_calculation = input("Do you want to perform another calculation? (yes/no) ")
        if next_calculation.lower() != 'yes':
            break

    print("Exiting Calculation, Goodbye!")

# Calling the calculator function
calculator()