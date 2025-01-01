# -----Developed By: Muhammad Umair Habib-----

# Import the regular expression module to handle pattern matching
import re

# A dictionary that maps keywords to predefined responses
responses = {
    "hello": "Hi, How can I assist you today?",
    "hi": "Hello!, How can I help you?",
    "how are you": "I'm just a bot, but I'm doing great! How are you?",
    "what is your name": "I'm a chatbot create to assist you.",
    "help": "Sure, I'm here to help you. What do you need to assistance with?",
    "bye": "Goodbye! Have a great day!",
    "thankyou": "You're welcome! I'm happy to help you",
    "default": "I'm not sure I understand. Could you please rephrase?"
}

# Function to define the appropriate response based on the user's input
def chatbbot_response(user_input):
    # Convert user input to lowercase to make matching case-sensitive
    input_user = user_input.lower()

    for keyword in responses:
        if re.search(keyword, user_input):
            return responses[keyword]

    return responses[default]

# Main function to run the chatbot
def chatbot():
    print("Chatbot: Hello! I'm here to assit you. (type 'bye' to exit)")

    while True:
        # Get user input
        user_input = input("You: ")

        # If user type 'bye', exit the loop
        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye! Have a great day!")
            break

        # Get chatbot's response based on user input
        response = chatbbot_response(user_input)

        # Print the Chatbot's response
        print("Chatbot: ", response)

# Run the chatbot
chatbot()