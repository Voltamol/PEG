def extract_story(text):
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start != -1 and end != -1:
        return text[start:end]
    return text

with open("Alice's adventures in Wonderland.txt") as iowrapper:
    txt=iowrapper.read()
    extracted=extract_story(txt)

with open("Alice's adventures in Wonderland.txt","w") as iowrapper:
    iowrapper.write(extracted)
    print("saved extracted text....")