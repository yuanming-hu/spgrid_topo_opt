import taichi as tc
import smtplib
import os
import socket
import atexit

gmail_sender = 'taichi.messager@gmail.com'
gmail_passwd = '6:L+XbNOp^'

def send_crash_report(message='Your task (@{}) has (failed).'.format(socket.gethostname()),
                      receiver=os.environ['TC_MONITOR_EMAIL']):
    TO = receiver
    SUBJECT = 'Report'
    TEXT = message

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_sender, gmail_passwd)

    BODY = '\r\n'.join(['To: %s' % TO,
                        'From: %s' % gmail_sender,
                        'Subject: %s' % SUBJECT,
                        '', TEXT])

    try:
        server.sendmail(gmail_sender, [TO], BODY)
    except:
        print('Error sending mail')

    server.quit()
    
    
def enable():
  register_call_back()

crashed = False
keep = []
    
def register_call_back():
  def email_call_back(_):
    global crashed
    crashed = True
    send_crash_report()
  keep.append(email_call_back)
  call_back = tc.function11(email_call_back)
  tc.core.register_at_exit(call_back)

  @atexit.register
  def at_exit():
    if not crashed:
      send_crash_report(message='Congratulations! Your task (@{}) has finished!'.format(socket.gethostname()))

if __name__ == '__main__':
  register_call_back()
  tc.core.trigger_sig_fpe()
