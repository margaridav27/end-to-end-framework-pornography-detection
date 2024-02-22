import datetime


def format_time(elapsed):
  '''
    Takes a time in seconds and returns a string hh:mm:ss
  '''
  
  elapsed_rounded = int(round((elapsed))) # Round to the nearest second
  return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss