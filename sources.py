"""Source registry for The Unofficial Guide (domain: student dorm/housing life).

Each entry mirrors a row in planning.md's Documents table. `kind` tells the
ingester how to extract the substantive text, because not every site stores its
content the same way:

  - "article"   : standard page; content lives in <article>/<p> tags.
  - "discourse" : Discourse forum (College Confidential). The page is rendered
                  by JavaScript, so we hit the built-in `.json` API instead.
  - "xenforo"   : XenForo forum (AnandTech). Posts live in .bbWrapper divs,
                  not <p> tags.
  - "homepage"  : landing page whose real content (per-dorm reviews) is
                  JS-rendered; only generic copy is reachable without a browser.
"""

SOURCES = [
    {
        "id": 1, "name": "college_dorm_reviews", "kind": "homepage",
        "label": "College Dorm Reviews",
        "url": "https://collegedormreviews.com",
    },
    {
        "id": 2, "name": "daily_pennsylvanian", "kind": "article",
        "label": "The Daily Pennsylvanian — dorm living tips",
        "url": "https://www.thedp.com/article/2016/06/new-student-issue-tips-dorm-living",
    },
    {
        "id": 3, "name": "cc_random_vs_chosen_roommate", "kind": "discourse",
        "label": "College Confidential — random vs. chosen roommate",
        "url": "https://talk.collegeconfidential.com/t/random-roommate-vs-choosing-roommate/1811434",
    },
    {
        "id": 4, "name": "cc_single_vs_roommate", "kind": "discourse",
        "label": "College Confidential — single vs. roommate",
        "url": "https://talk.collegeconfidential.com/t/single-vs-roommate-freshman-year/125580",
    },
    {
        "id": 5, "name": "anandtech_what_to_bring", "kind": "xenforo",
        "label": "AnandTech — things to bring to a freshman dorm",
        "url": "https://forums.anandtech.com/threads/things-that-must-be-brought-into-a-freshman-dorm.196580/",
    },
    {
        "id": 6, "name": "anandtech_roommates", "kind": "xenforo",
        "label": "AnandTech — roommates in college",
        "url": "https://forums.anandtech.com/threads/roommates-in-college.833171/",
    },
    {
        "id": 7, "name": "amherst_gear_reflection", "kind": "article",
        "label": "Amherst Student Blog — one year later gear reflection",
        "url": "https://admissionstudentblogs.wordpress.amherst.edu/?p=2911",
    },
    {
        "id": 8, "name": "purdue_dorm_advice", "kind": "article",
        "label": "Purdue Ambassador Blog — dorm life advice",
        "url": "https://ag.purdue.edu/agry/ambassadorblog/dorm-life-advice",
    },
    {
        "id": 9, "name": "grown_and_flown_seven_things", "kind": "article",
        "label": "Grown and Flown — seven things she wishes she'd known",
        "url": "https://grownandflown.com/student-wishes-she-had-known-before-freshman-year-college/",
    },
    {
        "id": 10, "name": "aol_tiktok_roommates", "kind": "article",
        "label": "In The Know / AOL — viral TikTok roommate experiences",
        "url": "https://www.aol.com/lifestyle/college-students-compare-freshman-dorm-183712927.html",
    },
]
