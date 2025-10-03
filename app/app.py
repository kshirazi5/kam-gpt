"""Streamlit entry point for the Kam-GPT portfolio app."""

from datetime import datetime
import streamlit as st


st.set_page_config(
    page_title="Kam-GPT",
    page_icon="🤖",
    layout="wide",
    menu_items={
        "Report a bug": "mailto:kamran@example.com",
        "About": "Kam-GPT is an AI-powered portfolio for Kamran Shirazi.",
    },
)


def render_header() -> None:
    """Render the hero section of the landing page."""
    st.title("Kam-GPT")
    st.subheader("An AI-powered portfolio for Kamran Shirazi")
    st.write(
        "This interactive profile showcases Kamran's experience, "
        "engineering projects, and AI capabilities."
    )


def render_highlights() -> None:
    """Render key professional highlights."""
    st.header("Highlights")
    cols = st.columns(3)
    highlights = [
        ("Experience", "6+ years building scalable data platforms."),
        ("Focus", "Machine learning, MLOps, and product engineering."),
        ("Location", "Remote-friendly, based in Toronto, Canada."),
    ]
    for col, (title, description) in zip(cols, highlights):
        with col:
            st.metric(label=title, value=description)


def render_timeline() -> None:
    """Render a simple career timeline."""
    st.header("Career Timeline")
    timeline = [
        ("2024", "Leading AI platform initiatives at ACME Co."),
        ("2022", "Launched data-driven growth experiments at Startup XYZ."),
        ("2019", "Scaled analytics stack for fintech clients."),
        ("2016", "Graduated with B.Sc. in Computer Science."),
    ]
    for year, event in timeline:
        st.markdown(f"**{year}** – {event}")


def render_footer() -> None:
    """Render footer information."""
    st.divider()
    current_year = datetime.now().year
    st.caption(f"© {current_year} Kamran Shirazi • Built with Streamlit")


def main() -> None:
    """Application entry point."""
    render_header()
    render_highlights()
    render_timeline()
    render_footer()


if __name__ == "__main__":
    main()
