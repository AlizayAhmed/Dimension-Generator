import os
import streamlit as st
from groq import Groq

# ------------------------------
# Local Physics Dictionary
# ------------------------------
DIMENSIONS = {
    "length": "L",
    "mass": "M",
    "time": "T",
    "speed": "LT^-1",
    "velocity": "LT^-1",
    "acceleration": "LT^-2",
    "force": "MLT^-2",
    "momentum": "MLT^-1",
    "energy": "ML^2T^-2",
    "power": "ML^2T^-3",
    "pressure": "ML^-1T^-2",
    "density": "ML^-3",
}

# ------------------------------
# Initialize Groq Client (if key is set)
# ------------------------------
def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not set in environment variables. AI fallback disabled.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq client: {e}")
        return None

# ------------------------------
# Query Groq for dimension inference
# ------------------------------
def get_ai_dimension(quantity, client):
    if not client:
        return None

    try:
        prompt = f"""
        You are a physics expert. Determine the dimensional formula for the physical quantity "{quantity}".
        Respond only with the dimensional expression in standard format (e.g., MLT^-2 for Force).
        """
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI lookup failed: {e}")
        return None

# ------------------------------
# Streamlit App UI
# ------------------------------
def main():
    st.set_page_config(page_title="Physics Dimension Calculator", page_icon="⚛️", layout="centered")
    st.title("⚛️ Physics Quantities → Dimension Calculator")
    st.caption("Hybrid Version: Uses local data + Groq AI fallback")

    quantity = st.text_input("Enter a physical quantity:", "").strip().lower()

    if st.button("Calculate"):
        if not quantity:
            st.warning("Please enter a quantity name.")
            return

        # Local lookup first
        if quantity in DIMENSIONS:
            dimension = DIMENSIONS[quantity]
            same_dimensions = [q for q, d in DIMENSIONS.items() if d == dimension and q != quantity]

            st.success(f"**Dimension of {quantity.capitalize()}:** {dimension}")
            if same_dimensions:
                st.info(f"**Quantities with same dimension:** {', '.join(same_dimensions)}")
            else:
                st.info("No other quantity shares this dimension.")

        else:
            st.warning("Quantity not found in local database. Trying AI inference... 🤖")
            client = get_groq_client()
            ai_dimension = get_ai_dimension(quantity, client)

            if ai_dimension:
                st.success(f"**AI Inferred Dimension of {quantity.capitalize()}:** {ai_dimension}")
            else:
                st.error("Unable to determine dimension. Please check your input or try again later.")

    st.markdown("---")
    st.caption("Developed by Alizay Ahmed")

# ------------------------------
if __name__ == "__main__":
    main()
