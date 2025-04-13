from string import Template
from textwrap import dedent
import streamlit as st


class HTML_Template:
    base_style = Template(
        """
        <style>
            $css
        </style>
        """
    )


class MainCSS:
    initial_page_styles = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        /* General styles */
        body {
            font-family: 'Inter', sans-serif; /* Consistent font */
        }
        #GithubIcon {
            visibility: hidden;
        }

        .stApp {
            padding: 2rem; /* Consistent padding */
        }

        /* Sidebar */
        .stSidebar {
            padding: 1rem; /* Maintain padding */
            transition: width 0.3s ease-in-out; /* Smooth sidebar expansion */
        }
        .stSidebar:hover {
            width: 270px; /* Expand on hover */
        }

        /* Buttons */
        .stButton>button {
            border-radius: 8px; /* Rounded corners */
            font-weight: bold; /* Bold text */
            padding: 12px; /* Padding for usability */
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease; /* Smooth hover effect */
        }
        .stButton>button:hover {
            filter: brightness(85%); /* Universal darken effect */
        }

        /* Typography */
        .stMarkdown h1 {
            font-size: 28px; /* Main header */
            font-weight: 700;
        }
        .stMarkdown h2 {
            font-size: 24px; /* Subheader */
            font-weight: 600;
        }
        .stMarkdown p {
            font-size: 16px; /* Paragraphs */
            font-weight: 400;
        }

        /* Justify text for Markdown, info, and general text */
        .stMarkdown p, .stMarkdown div, .stAlert {
            text-align: justify;
        }

    """.strip()
