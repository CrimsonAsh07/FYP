import datetime
import pywhatkit as kit

def send_whatsapp_message(phone_number, message):
    try:
        now = datetime.datetime.now()
        kit.sendwhatmsg(f"+{phone_number}", message, now.hour, now.minute + 1)  
        print("Message sent successfully!")
        return True
    except Exception as e:
        print("Error sending Alert:", str(e))
        return False

# Uncomment to test function
# if __name__ == "__main__":
#     recipient_number = 'NUMBER' NUMBER with Country Code without +
#     message = "Hello from your Python script!"

#     send_whatsapp_message(recipient_number, message)
