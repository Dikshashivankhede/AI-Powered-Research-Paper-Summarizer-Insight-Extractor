import streamlit as st

st.set_page_config(layout="wide")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm_handler import ask_llm

from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components
import pandas as pd
import io

# ================= UI STYLING =================
st.markdown("""
<style>

/* BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* 3D HEADING */
h1 {
    text-align: center;
    font-size: 48px;
    color: #00FFFF;
    text-shadow: 0 0 10px #00FFFF, 0 0 25px #00C9FF;
}

/* CARD */
.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}

/* INPUT */
.stTextInput input {
    background-color: rgba(255,255,255,0.1);
    color: white;
    border-radius: 12px;
    border: 2px solid #00FFFF;
    padding: 12px;
}

/* BUTTON */
.stButton button {
    background: linear-gradient(90deg, #00FFFF, #00C9FF);
    color: black;
    border-radius: 20px;
    font-weight: bold;
    padding: 10px 25px;
}

/* 🔥 TABS FIX (HIGH CONTRAST) */
.stTabs [role="tab"] {
    background: #1e1e1e;
    color: #cccccc;
    border-radius: 10px;
    padding: 10px 20px;
    margin-right: 8px;
}

.stTabs [aria-selected="true"] {
    background: #00FFFF;
    color: black !important;
    font-weight: bold;
}

/* 🔥 METRIC CARDS */
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}
.metric-value {
    font-size: 28px;
    color: #00FFFF;
    font-weight: bold;
}
.metric-label {
    font-size: 14px;
    color: #ccc;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown("""
<h1>🚀📄 AI-Powered Research Paper Summarizer & Insight Extractor</h1>
<p style='text-align:center; color:#00FFFF;'>
Ask questions • Explore insights • Visualize knowledge graphs
</p>
""", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2 = st.tabs(['📄 Research Paper QA', "🧠 Knowledge Graph Explorer"])

# =====================================================
# TAB 1 (UNCHANGED LOGIC)
# =====================================================
with tab1:

    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_db = FAISS.load_local(
            "research_papers_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vector_db

    vector_db = load_vector_db()

    st.markdown(f"""
    <div class="card">
    <h3>📚 Total Papers in Database: {vector_db.index.ntotal}</h3>
    </div>
    """, unsafe_allow_html=True)

    user_query = st.text_input("🔎 Ask a question about research papers:")

    if st.button("🚀 Get Insights"):

        results = vector_db.similarity_search(user_query, k=3)

        content = ""

        for idx, doc in enumerate(results, 1):
            title = doc.metadata.get("title", f"Paper {idx}")

            content += f"""
            Paper Title: {title}

            Paper Content:
            {doc.page_content}
            """

        with st.spinner("🤖 AI is analyzing papers..."):
            response = ask_llm(content, user_query)

        # -------- SAME LOGIC --------
        answer = ""
        paper_titles = []

        if "Research Paper:" in response:
            parts = response.split("Research Paper:")
            answer = parts[0].replace("Answer:", "").strip()
            papers_text = parts[1].strip()
            paper_titles = [p.strip() for p in papers_text.split(",")]
        else:
            answer = response.strip()

        st.markdown("<div class='card'><h3>🤖 AI Insight</h3></div>", unsafe_allow_html=True)
        st.write(answer)

        if paper_titles and "none" not in [p.lower() for p in paper_titles]:

            st.markdown("<div class='card'><h3>📄 Relevant Research Papers</h3></div>", unsafe_allow_html=True)

            for doc in results:
                title = doc.metadata.get("title", "")

                for p in paper_titles:
                    if title.lower() == p.lower():
                        with st.expander(f"📄 {title}"):
                            st.write(doc.page_content)

        else:
            st.warning("No relevant research paper found.")

# =====================================================
# TAB 2 (UNCHANGED LOGIC + UI ONLY)
# =====================================================
with tab2:

    st.markdown("<div class='card'><h2>🧠 Knowledge Graph Explorer</h2></div>", unsafe_allow_html=True)

    @st.cache_resource
    def get_driver():
        return GraphDatabase.driver(
            "neo4j://127.0.0.1:7687",
            auth=('neo4j','myinstance123')
        )

    driver = get_driver()

    @st.cache_data
    def get_domain():
        query = """ 
        MATCH (d:Domain)
        RETURN d.name as domain
        """
        with driver.session() as session:
            result = session.run(query)
            domains = [r["domain"] for r in result]

        normalized = {}
        for d in domains:
            normalized[d.lower()] = d.title()

        return sorted(normalized.values())

    domain = st.selectbox("Select Research Domain", get_domain())

    def get_graph_data(domain):
        query = """ 
        MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
        WHERE toLower(d.name) = toLower($domain)
        
        OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
        OPTIONAL MATCH (p)-[:USES]-(m:Method)
        
        RETURN p.title AS paper,
        a.name AS author,
        m.name AS method,
        d.name AS domain
        """
        with driver.session() as session:
            result = session.run(query, domain=domain)
            return [r.data() for r in result]

    def draw_graph(data):
        net = Network(height="600px", width="100%", bgcolor="#111111", font_color="white")

        for row in data:
            paper = row['paper']
            author = row['author']
            method = row['method']
            domain = row['domain']

            net.add_node(paper, label=paper, color="orange")

            if author:
                net.add_node(author, label=author, color="skyblue")
                net.add_edge(author, paper)

            if method:
                net.add_node(method, label=method, color="green")
                net.add_edge(paper, method)

            if domain:
                net.add_node(domain, label=domain, color="purple")
                net.add_edge(paper, domain)

        net.save_graph("graph.html")

        with open("graph.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=600)

    if domain:
        st.subheader(f"Knowledge Graph for Domain: {domain}")  

        data = get_graph_data(domain)

        if len(data) == 0:
            st.warning("No papers found")

        else:
            df = pd.DataFrame(data)

            # 🔥 SAME LOGIC, ONLY UI CHANGED
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df['paper'].nunique()}</div>
                    <div class="metric-label">Total Papers</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df['author'].nunique()}</div>
                    <div class="metric-label">Total Authors</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{df['method'].nunique()}</div>
                    <div class="metric-label">Total Methods</div>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            st.subheader("Filtered Research Data")
            st.dataframe(df, use_container_width=True)

            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)

            st.download_button(
                "📥 Download Excel",
                excel_buffer,
                file_name=f"{domain}_research_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.divider()
            draw_graph(data)