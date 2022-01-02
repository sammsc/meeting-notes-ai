import sys
import re
from pathlib import Path
import pandas as pd
from transformers import pipeline
from nltk.tokenize import RegexpTokenizer
import nltk.data

import streamlit as st
from io import StringIO
import plotly.express as px
# import streamlit.components.v1 as components
import os
# import time

nltk.download('punkt')


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def fill_df(speaker_id, speaker_time, txt_orig):
    global df_transcript
    # txt_summary = summarizer(txt_orig, max_length=90, min_length=25, do_sample=False)
    txt_summary = ''
    df_transcript = df_transcript.append(pd.Series(
        [speaker_id, speaker_time, txt_orig, txt_summary],
        index=['speaker', 'time', 'text', 'summary']),
        ignore_index=True)


def split_analyze(txt_input, char_lim=1000, frac_overlap=0.0, model='summary'):
    '''
    Split long text into chunks of sentences, with each chunk not exceeding number
    of characters in char_lim. Each chunk is variable in length since the split is
    done by sentence not character. Successive chunks can overlap each other, with
    amount of overlap ranging from 0 (no overlap) to 1 (complete overlap). Each chunk
    is passed into summarizer to get its summary. Text is parsed from bottom to top
    order. The summaries are concatenated into one string as the return value.

    txt_input: a single string to be summarized
    char_lim: the upper limit of characters for each chunk
    frac_overlap: the fraction of overlap between successive chunks
    '''

    if char_lim < 0:
        char_lim = 0
    if frac_overlap < 0:
        frac_overlap = 0
    elif frac_overlap > 0.8:
        frac_overlap = 0.8

    char_overlap = char_lim * frac_overlap
    sentence_list = tokenizer.tokenize(txt_input)
    sentence_list_rev = list(reversed(sentence_list))
    sentence_ttl = len(sentence_list_rev)
    char_ct = 0
    summary_input = []
    summary_output = []
    temp_sentence_rev = []

    def get_summary(list_sentence_rev):
        cur_input = ' '.join(reversed(list_sentence_rev))
        summary_input.append(cur_input)

        # print(f'{model}, {len(list_sentence_rev)}, {len(cur_input)}')

        if model == 'summary':
            cur_output = summarizer(cur_input, max_length=90, min_length=25, do_sample=False)
            summary_output.append(cur_output[0]['summary_text'])
        elif model == 'sentiment':
            cur_output = classifier_total(cur_input)
            summary_output.append(cur_output[0])
        else:
            sys.exit("model should be summarizer or sentiment")

    lp_ttl = len(sentence_list_rev)
    # printProgressBar(0, lp_ttl, prefix='Progress:', suffix='Complete', length=50)

    c1.write(f'running {model} analysis')
    my_bar = c1.progress(0)

    for sen_idx, sen_cur in enumerate(sentence_list_rev):
        char_ct += len(sen_cur)
        temp_sentence_rev.append(sen_cur)

        if (sen_idx + 1) < sentence_ttl:
            if (char_ct + len(sentence_list_rev[sen_idx + 1])) > char_lim:
                get_summary(temp_sentence_rev)
                while char_ct > char_overlap:
                    sen_del = temp_sentence_rev.pop(0)
                    char_ct -= len(sen_del)

        # Update Progress Bar
        # printProgressBar(sen_idx + 1, lp_ttl, prefix='Progress:', suffix='Complete', length=50)
        my_bar.progress(sen_idx / lp_ttl)

    get_summary(temp_sentence_rev)

    if model == 'summary':
        final_output = ' '.join(reversed(summary_output))
        return final_output
    elif model == 'sentiment':
        box_positive = 0
        box_negative = 0
        for i in summary_output:
            if i['label'] == 'POSITIVE':
                box_positive += 1
            else:
                box_negative += 1
        # print(box_positive)
        # print(box_negative)
        if (box_negative / (box_positive + box_negative)) < 0.65:
            return 'positive'
        else:
            return 'negative'


# config streamlit
st.set_page_config(layout="wide")
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

padding = 0.5
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

c1, c2, c3 = st.columns([1, 3, 4])


# Read in transcript
with c1:
    # st.title('meeting-notes')
    st.image('logo.png')
    uploaded_file = st.file_uploader('upload zoom meeting transcript',
                                     key='file_upload',
                                     help='upload zoom meeting transcript')

if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    text_single = stringio.read()

    text_all = text_single.split('\n')
    # print(f'text_all: type is {type(text_all)}, length is {len(text_all)}')

    # Setup summarization, sentiment-analysis model,
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    classifier_total = pipeline('sentiment-analysis')

    df_transcript = pd.DataFrame(columns=['speaker', 'time', 'text', 'summary'])


    # Parse transcript into dataframe by speaker segment
    first_line = True
    txt_orig = ''
    speaker_id = ''
    speaker_time = ''
    for line_idx, line_cur in enumerate(text_all):
        if len(line_cur.strip()) != 0:
            # matching speaker ID line
            m_res = re.search(r'^Speaker (?P<id>\d+) \((?P<time>\d+:\d+)\):\s*\Z', line_cur)
            if m_res:
                if first_line:
                    first_line = False
                else:
                    fill_df(speaker_id, speaker_time, txt_orig)
                speaker_id = m_res.group('id')
                speaker_time = m_res.group('time')
                txt_orig = ''
            else:
                txt_orig += line_cur.rstrip('\n')

    fill_df(speaker_id, speaker_time, txt_orig)

    # Group text by speaker
    df_txt_all = df_transcript.groupby(['summary'])['text'].apply(' '.join).reset_index()
    df_by_speaker = df_transcript.groupby(['speaker'])['text'].apply(' '.join).reset_index()
    txt_all = df_txt_all.loc[0, 'text']

    # Calculate speaker stats
    tokenizer = RegexpTokenizer(r'\w+')
    word_count = []
    for row in df_by_speaker.itertuples():
        tokens = tokenizer.tokenize(row.text)
        word_count.append(len(tokens))

    df_by_speaker['word_count'] = word_count
    df_by_speaker['word_percent'] = (df_by_speaker['word_count'] /
                      df_by_speaker['word_count'].sum()) * 100
    df_speak_counts = df_transcript.groupby(['speaker']).size().reset_index(name='interactions')
    df_by_speaker = df_by_speaker.merge(df_speak_counts, how='outer', on='speaker')
    fig = px.pie(df_by_speaker, values='word_percent', names='speaker', hover_data=['interactions'])

    # Setup nltk tokenizer for splitting text by sentence
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    summary_all = split_analyze(txt_all, char_lim=2000, frac_overlap=0.1)
    summary_of_summaries = split_analyze(summary_all, char_lim=2000, frac_overlap=0.1)
    sentiment_score_draft = split_analyze(txt_all, char_lim=100, model='sentiment')

    with c2:
        st.subheader('Summary')
        st.text_area(label='', value=summary_of_summaries, height=300)
        st.subheader('Transcript')
        st.text_area(label='', value=txt_all, height=500)

    with c3:
        st.subheader('Participation')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Meeting Sentiment')
        st.write(sentiment_score_draft)

