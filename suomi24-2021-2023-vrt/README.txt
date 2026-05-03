Resource title (English): The Suomi24 Corpus 2021-2023, VRT version

Resource title (Finnish): Suomi24-korpus 2021-2023, VRT versio

Shortname: suomi24-2021-2023-vrt

Metadata: http://urn.fi/urn:nbn:fi:lb-2024121301

Rightholder: City Digital Group

License: ACA-NC
The complete license is available at http://urn.fi/urn:nbn:fi:lb-2022020821

A copy of the license is included in LICENSE.txt. The license details
may be subject to change, so before downloading the resource, please
refer to the latest version of the license at the above link.

Resource group page: http://urn.fi/urn:nbn:fi:lb-2022011221



Short description

The corpus contains all the texts available in the discussion forums
of the Suomi24 online social networking website from 1 January 2021 to
31 December 2023. The data was tokenized, converted to VRT format and
annotated at the Language Bank of Finland.

The entire corpus in the VRT format is downloadable for academic
research purposes.


Detailed description

This data set is an annotated VRT version of a full database dump of
the content of the Suomi24 discussion forums
(https://keskustelu.suomi24.fi) from 1 January 2021 to 31 December
2023 from City Digital Group, received in March 2024. The data set
excludes data from closed or hidden discussion topics.

The data was cleaned up, tokenized, transformed to the VRT format and
morpho-syntatically annotated for FIN-CLARIN in the CSC Puhti
environment with ad-hoc and FIN-CLARIN VRT Tools scripts running e.g.
the UDPipe tokenizer (finnish-tdt model, with post-processors) and the
already old dependency analysis tools and models (TDPP) from Turku NLP
group (adapted for VRT in the language bank, models used as they
were). In addition, names were recognized by the FiNER tagger, a part
of Finnish Tagtools 1.6 (http://urn.fi/urn:nbn:fi:lb-2024021401),
sentence languages were identified with HeLI-OTS 2.0
(https://urn.fi/urn:nbn:fi:lb-2024040301), and sentence sentiment
polarity was annotated by a sentiment classifier trained on the
FinnSentiment corpus (see https://arxiv.org/pdf/2012.02613.pdf). The
messages were then reordered and augmented with derived attributes.

The data has been divided into files by the year, corresponding to the
subcorpora in Korp. The messages within each year are sorted by
thread, and threads are sorted by the timestamp of the first message
of the thread. Messages within a thread are sorted in thread order:
each message is followed by the direct comments to it (recursively),
sorted by their timestamp. Threads that span over several years have
been split by the year.

Messages appear as text elements that contain paragraph elements that
contain sentence elements that contain a sequence of annotated tokens.
Thread titles appear both as an attribute in each message and as a
paragraph in the first message of the thread.

The text elements contain the following essential attributes:
- msg_type: "thread_start" or "comment"
- thread_id: thread identifier (number)
- comment_id: comment identifier (number; 0 if thread start message)
- msg_id: constructed message id (thread_id:comment_id)
- parent_comment_id: parent-comment identifier (0 if thread start
  message or if parent is the thread start message)
- quoted_comment_id: quoted-comment identifier (0 if no quotation)
- date: creation date (2019-01-11)
- time: creation time (16:55:26)
- datetime: combined creation date and time (2019-01-11 16:55:26)
- thread_start_datetime: creation date and time of the thread start
  message (2018-01-01 01:30:00)
- parent_datetime: creation date and time of the parent comment
  (2018-01-01 01:30:00, empty for thread start messages)
- author: user nickname
- author_logged_in: whether author was logged in (y, n)
- title: thread title from starting message
- topic_names: hierarchical topic (discussion area) name, top level
  first, levels separated by " &gt; " ("Ajoneuvot ja liikenne &gt;
  Autot &gt; Automerkit &gt; Honda")
- topic_names_set: topic level names as a set ("|Ajoneuvot ja
  liikenne|Automerkit|Autot|Honda|")
- topic_name_top: top-level topic name ("Ajoneuvot ja liikenne")
- topic_name_leaf: bottom-level topic name ("Honda")
- topic_adultonly: whether the topic is for adults only (y, n)
- thread_closed: whether the thread is closed (new comments cannot be
  written) (y, n)
- empty: whether the original message was completely empty (y, n)
- sum_lang: the ISO 639-3 codes of languages identified in the
  sentences of the text and the number of sentences in each language
  (see the sentence attribute lang below for some more information)
  ("|fin:37|izh:1|und:1|", ordered by number of occurrences, tied
  codes alphabetically)
- id: unique text identifier (t-b65f2a5b-f0bc7713)

The following text element attributes can be derived from other
attributes, are included mostly for backward-compatibility or are
otherwise less essential:
- title_orig: original title with possible leading, trailing and
  multiple consecutive spaces preserved
- topic_names_orig: original hierarchical topic (discussion area)
  name, possibly with misordered hierarchy levels and double spaces
- datetime_approximated: whether the date and time were approximated
  based on the surrounding messages (always "n" (no) in this data)
- author_nick_registered: whether nickname was registered (always the
  same as the value of "author_logged_in")
- user_id: a user identifier of 32 hexadecimal digits for logged-in
  users, corresponding to the nickname in attribute author; 0 for
  others
- hierarchy_id: an id (number) in the comment messages of the original
  data whose purpose is unknown to us but kept just in case; empty for
  thread start messages
- datefrom, dateto: creation date (20190111)
- timefrom, timeto: creation time (165526)
- author_name_type: always "user_nickname"
- filename_vrt: the name of the VRT file containing the message during
  processing
- filename_orig: the name of the VRT file containing the message at
  the beginning of the processing
- origfile_textnum: the number of the corresponding text element in
  the VRT file indicated by "filename_orig" (1-based)
- _sort_key: the key according to which the messages were sorted
  (byte-wise) within each thread

Paragraph attributes:
- type: "title" or "body"
- sum_lang: the ISO 639-3 codes of languages identified in the
  sentences of the paragraph and the number of sentences in each
  language (see the sentence attribute lang below for some more
  information) ("|fin:2|und:1|")
- id: unique paragraph identifier (p-b65f2a5b-6aae814a-5288b71c)

Sentence attributes:
- lang: ISO 639-3 code of the language of the sentence as identified
  by HeLI-OTS 2.0; "und" for non-language data
- lang_conf: a confidence value of the language identification
  provided by HeLI-OTS
- sentiment_polarity: sentiment polarity of the sentence: "pos",
  "neut" or "neg"
- id: unique sentence identifier (s-b65f2a5b-f0bc7713-2ccba717)
- _skip: "|finnish-nertag|" if the sentence was not annotated with
  names; completely missing otherwise ("|" in Korp)

In addition to these elements to which all tokens belong, name (and
time and number) expressions recognized by FiNER 1.6 are enclosed in
"ne" elements with the following attributes:
- name: the name enclosed by the element, possibly multi-word; for
  name expressions, the last word is the base form of the last token,
  whereas the preceding ones are word forms
- fulltype: the complete type of the name as recognized by FiNER
  ("EnamexOrgCrp")
- ex: the main category of the expression: "ENAMEX" (name), "TIMEX"
  (time expression) or "NUMEX" (numerical expression)
- type: the broad type of the expression ("ORG")
- subtype: the finer type of the expression ("CRP")
- placename: same as the value for "name" if the name is recognized as
  a place name, empty otherwise
- placename_source: "ner" if the name is recognized as a place name,
  empty otherwise

Nested name expressions are enclosed in "ne1" and "ne2" elements with
the same attributes as "ne". "ne1" elements occur only within "ne" and
"ne2" only within "ne1".

The order of the attributes in the element start tags is arbitrary but
fixed.

The original data contained six completely empty messages. To preserve
their information in the VRT data, a lone underscore was added as
their content, with the appropriate annotations. The attribute "empty"
of these texts have the value "y".

The first line of each VRT file is a special comment that names the
positional attributes (tab-separated fields) in order:

<!-- #vrt positional-attributes: word ref lemma lemmacomp pos msd dephead deprel spaces initid lex/ nertag2 nertags2/ nerbio2 -->

- word: surface form of the token
- lemma: base form
- lemmacomp: base form with compound-boundary markers (vertical bars)
  separating compound parts
- pos: part of speech
- msd: morpho-syntactic description
- ref: the number of the token in the sentence
- dephead: dependency head number (0 if no head)
- deprel: dependency relation
- spaces: spaces around (or within) the token in the original data
  (from tokenizer)
- initid: running number (from tokenizer; largely redundant with ref)
- lex/: lemgram, a combination of base form and a part-of-speech tag,
  surrounded by vertical bars
- nertag2: maximal name information produced by FiNER, of the form
  CategoryTypSbt-X, where CategoryTypSbt is the full type of the name
  (see above) and X is one of "B" (the first word of a multi-word
  name), "E" (the last word of a multi-word name) or "F" (a
  single-word name)
- nertags2/: name information produced by FiNER, including possible
  nested names: values CategoryTypSbt-X-N separated by vertical bars,
  where CategoryTyp and X are as in "nertag2" and N is the nesting
  level (0, 1 or 2), with 0 being the outermost (maximal) name
- nerbio2: a different kind of name information produced by FiNER for
  maximal names: B-TYP (the first word of a name of with broad type
  TYP), I-TYP (a subsequent word of a name with type TYP) or O
  (outside a name)

Since the parser produced some multi-rooted analyses anyway, the long
sentences that were parsed in shorter shreds were left multi-rooted
when the shreds were put back together.

The three characters < > & appear as &lt; &gt; &amp; everywhere
(because in bare form they are used for the markup), and the double
quotation mark " appears as &quot; in text attribute values. Attribute
values are always enclosed in double quotation marks.

Otherwise all content is encoded as UTF-8. Spurious control characters
were interpreted or removed, space characters were normalized and
non-characters were removed. However, Unicode normalization was not
done, nor ligatures considered; unassigned code points and private-use
characters may have been handled.

No attempt was made to normalize the various characters used or abused
for quotation marks, apostrophes, or dashes.

Over-long "words" were shortened and marked with "REDACTED" in the
data, partly for processing reasons.

Sentences containing certain kinds of constructs, such as longish
sequences of consecutive all-uppercase words, were not annotated with
names for processing reasons.

Each VRT file contains a couple of informational XML-style comment
lines ("<!-- ... -->") at the beginning and end of the file.


Differences from earlier parts of Suomi24 data

The format of the data of Suomi24 2021–2023, VRT version is mostly the
same as that of Suomi24 2018–2020, VRT version 1.1
(http://urn.fi/urn:nbn:fi:lb-2020021801). The few differences are due
to differences in the original source data or in processing the data.

Missing text attribute:
- author_orig: same as "author" in Suomi24 2018–2020, but added for
  compatibility with Suomi24 2001–2017, where it is the original
  author nickname that may contain leading, trailing or multiple
  consecutive spaces, normalized to attribute "author"; in Suomi24
  2021–2023, values of "author" did not have spurious spaces

Missing sentence attributes:
- polarity: renamed to the more descriptive "sentiment_polarity" and
  retained in Suomi24 2018–2020 as an alias for
  backward-compatibility
- lang_v1: the code for the language identified for the sentence by
  HeLI-OTS 1.1 ("lang" in Suomi24 2018–2020, VRT version 1.0); can
  differ from that identified by HeLI-OTS 2.0

The values of "id" attributes are unique identifiers composed of
pseudo-random parts, whereas in Suomi24 2018–2020, the text "id" is
the same as "msg_id", and paragraph and sentence ids are running
numbers of the elements within the subcorpus (file).

In Suomi24 2021–2023, a token boundary has been added after a
punctuation mark immediately followed by an upper-case letter (???),
whereas in 2018–2020, such tokens have not been split, also resulting
in missing sentence breaks.

In addition to the above differences from Suomi24 2018–2020, the data
of Suomi24 2021–2023, VRT version differs from that of Suomi24
2001–2017, VRT version 1.3 (http://urn.fi/urn:nbn:fi:lb-2020021803) in
the following ways.

Additional text attributes (see above for their meaning):
- hierarchy_id
- thread_closed
- user_id

Some attributes are missing as either the information they contained
was not available in the source data or they were not applicable to
this data:
- author_v1: not applicable
- author_nick_type: whether nickname was registered (information not
  in source data)
- author_signed_status: whether nickname was registered and the author
  logged in (nickname registration status not in source data)
- topic_nums: comma-separated topic numbers (not in source data)
- topic_nums_set: topic numbers as a set (not in source data)

Values for the positional attribute "lemma" for compound words may
differ from those in Suomi24 2001–2017, as they are intended to be
more natural, without lemmatizing all compound parts of the word.


For further information, please contact fin-clarin@helsinki.fi .
