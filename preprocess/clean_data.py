import re

class Cleaner:
    def __init__(self, text):
        self.txt = text
        self.clean_data = None
        
    def Clean_Resume(self):
        def clean_func(txt):
            cleantxt = re.sub('http\S+\s', ' ', txt)
            cleantxt = re.sub('RT|CC', ' ', cleantxt)
            cleantxt = re.sub('#\S+\s', ' ', cleantxt)
            cleantxt = re.sub('@\S+', ' ', cleantxt)
            cleantxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', cleantxt)
            cleantxt = re.sub(r'[^\x00-\x7f]', ' ', cleantxt)
            cleantxt = re.sub('\s+', ' ', cleantxt)
            self.clean_data = cleantxt
            return self.clean_data       
        self.clean_data = clean_func(self.txt)
        return self.clean_data
    