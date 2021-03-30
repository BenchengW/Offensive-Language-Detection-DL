###############################################################################################################
#these are preprocess function
#################################################################################################################
#load data
def load_data():
  raw_data_df = pd.read_csv('https://query.data.world/s/twuhmzuhvitwqqcjh5picrq3qykr4r')
  return raw_data_df

def hashtag(text):
  FLAGS = re.MULTILINE | re.DOTALL
  text = text.group()
  hashtag_body = text[1:]
  if hashtag_body.isupper():
      result = "<hashtag> {} <allcaps>".format(hashtag_body.lower())
  else:
      result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
  return result

def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps> " # amackcrane added trailing space
    # function so code less repetitive

def clean_data(text):
  FLAGS = re.MULTILINE | re.DOTALL
  eyes = r"[8:=;]"
  nose = r"['`\-]?"
  def re_sub(pattern, repl):
      return re.sub(pattern, repl, text, flags=FLAGS)

  text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
  text = re_sub(r"@\w+", "<user>")
  text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
  text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
  text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
  text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
  text = re_sub(r"/"," / ")
  text = re_sub(r"<3","<heart>")
  text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
  text = re_sub(r"#\w+", hashtag)  # amackcrane edit
  text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
  text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    

  ## -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
  # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
  #text = re_sub(r"([A-Z]){2,}", allcaps)  # moved below -amackcrane

  # amackcrane additions
  text = re_sub(r"([a-zA-Z<>()])([?!.:;,])", r"\1 \2")
  text = re_sub(r"\(([a-zA-Z<>]+)\)", r"( \1 )")
  text = re_sub(r"  ", r" ")
  text = re_sub(r" ([A-Z]){2,} ", allcaps)
    
  return text.lower()

def preprocessing_tweet(tweet_df):
  temp_list= []
  for t in tweet_df['tweet']:
    temp_list.append(clean_data(t))
  tweet_df['clean_tweet'] = temp_list
  return tweet_df
###########################################################################################################



