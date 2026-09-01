import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ⚠️ ضع هنا مفتاح جوجل السري الخاص بك بدقة
MY_SECRET_KEY = "AQ.Ab8RN6JKvG_nd75fLjZ_Qi7J88uBS_ih8BFi6DIYLdEU7MIiSw"

class ElectricChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("مستشار الهندسة الكهربائية الذكي - نظام المحادثة الشات v2.5")
        self.root.geometry("680x680") # زيادة العرض قليلاً لراحة القراءة
        self.root.configure(bg="#f4f6f9")
        
        self.chat_history = []
        self.retriever = None

        # 1. شريط العنوان العلوي الفخم
        title_frame = tk.Frame(root, bg="#1e3d59", pady=12)
        title_frame.pack(fill=tk.X)
        title_label = tk.Label(
            title_frame, text="⚡ نظام المشورة الفنية والدردشة الهندسية المنمقة 💬", 
            font=("Segoe UI", 12, "bold"), bg="#1e3d59", fg="white"
        )
        title_label.pack()

        # 2. منطقة عرض الشات
        chat_frame = tk.Frame(root, bg="#f4f6f9")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.chat_display = tk.Text(chat_frame, font=("Segoe UI", 11), bg="white", bd=0, wrap=tk.WORD, state=tk.DISABLED, spacing2=4)
        self.chat_display.tag_configure("user_msg", justify="right", foreground="#2b2b2b", spacing1=8, font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_configure("bot_msg", justify="right", foreground="#1e3d59", spacing1=8)
        self.chat_display.tag_configure("system_alert", justify="center", foreground="#777777", font=("Segoe UI", 10, "italic"))

        scrollbar = ttk.Scrollbar(chat_frame, command=self.chat_display.yview)
        self.chat_display.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.chat_display.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 3. شريط إدخال الأسئلة السفلي
        bottom_frame = tk.Frame(root, bg="#f4f6f9", pady=10)
        bottom_frame.pack(fill=tk.X, padx=15)
        
        self.send_btn = ttk.Button(bottom_frame, text="إرسال الدعم", command=self.start_chat_thread)
        self.send_btn.pack(side=tk.LEFT, padx=5)
        
        self.query_entry = tk.Entry(bottom_frame, font=("Segoe UI", 12), justify="right", bd=1, relief=tk.SOLID)
        self.query_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)
        self.query_entry.bind("<Return>", lambda event: self.start_chat_thread())

        self.append_to_display("⚙️ جاري فحص مراجع الهندسة الكهربائية وتهيئة خوارزميات الصياغة المنمقة...", "system_alert")
        threading.Thread(target=self.prepare_knowledge_base, daemon=True).start()

    def append_to_display(self, text, tag_name):
        self.chat_display.config(state=tk.NORMAL)
        if tag_name == "user_msg":
            self.chat_display.insert(tk.END, f"\n 👤 المهندس زكريا: {text}\n", tag_name)
        elif tag_name == "bot_msg":
            self.chat_display.insert(tk.END, f"\n🤖 الرد الاستشاري المنمق:\n{text}\n", tag_name)
            self.chat_display.insert(tk.END, "──────────────────────────────────────────────────\n", "system_alert")
        else:
            self.chat_display.insert(tk.END, f"\n{text}\n", tag_name)
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)

    def prepare_knowledge_base(self):
        try:
            loader = DirectoryLoader('./', glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            
            if not documents:
                self.root.after(0, lambda: self.append_to_display("❌ لم يتم العثور على أي ملفات مراجع هندسية PDF!", "system_alert"))
                return

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)

            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview", google_api_key=MY_SECRET_KEY)
            db = Chroma.from_documents(texts, embeddings)
            self.retriever = db.as_retriever(search_kwargs={"k": 4})
            
            self.root.after(0, lambda: self.append_to_display("✅ تم تحميل الفهرس الفني بنجاح! المستشار جاهز الآن للرد بصياغة احترافية.", "system_alert"))
        except Exception as e:
            self.root.after(0, lambda: self.append_to_display(f"❌ خطأ تقني: {e}", "system_alert"))

    def start_chat_thread(self):
        user_query = self.query_entry.get().strip()
        if not user_query:
            return
        
        if self.retriever is None:
            messagebox.showwarning("تنبيه", "برجاء الانتظار ثوانٍ حتى تكتمل التهيئة المعرفية.")
            return

        self.append_to_display(user_query, "user_msg")
        self.query_entry.delete(0, tk.END)
        self.send_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self.get_bot_response, args=(user_query,), daemon=True).start()

    def get_bot_response(self, user_query):
        try:
            docs = self.retriever.invoke(user_query)
            context = "\n\n".join(doc.page_content for doc in docs)

            # 🌟 تعليمات سحرية جديدة كلياً تدمج بين "جمالية اللسان العربي" و"دقة الأرقام الهندسية الصارمة"
            system_instruction = (
                "أنت مهندس استشاري أول وخبير رائد في مجالات الطاقة الشمسية، المحولات، والمولدات الكهربائية. "
                "مهمتك هي صياغة إجابة فنية منمقة للغاية، بليغة، ومنظمة تجذب القارئ وتمنحه الثقة المطلقة في جودة التطبيق.\n\n"
                "⚠️ قواعد التنسيق والصياغة الإلزامية:\n"
                "1. استخدم لغة عربية فصحى قوية، رصينة، ومهنية جداً بأسلوب استشاري رفيع.\n"
                "2. رتّب الإجابة بصرياً بشكل رائع باستخدام الرموز التعبيرية الوظيفية (مثل: 🔹 للعناصر، ⚡ للتنبيهات الكهربائية، 📊 للحسابات، 🛠️ للصيانة).\n"
                "3. قسّم النصوص الطويلة إلى فقرات قصيرة ومنظمة، واجعل العناوين الفرعية واضحة وبخط منمق.\n"
                "4. ادخل في التفاصيل الفنية الدقيقة المستخرجة من المراجع المتاحة، واذكر الأرقام والمعادلات بوضوح تام وبدون اختصار مخل.\n"
                "5. وثّق إجابتك بذكر المصدر أو اسم الدليل المرفق في نهاية الإجابة إن وجد لتعزيز المصداقية.\n"
                "6. إذا كانت البيانات مفقودة في المراجع، اعتذر بأسلوب راقٍ ومنمق كالتالي: 'نود الإفادة بأن هذه الجزئية الفنية تقع خارج نطاق المراجع المعتمدة لدينا حالياً'.\n\n"
                "المراجع الهندسية المتاحة للاستنباط:\n{context}"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])

            # تم ضبط الـ temperature على 0.3 لإعطاء النموذج مرونة اختيار مرادفات عربية فخمة ومنمقة مع الحفاظ الكامل على دقة المعلومة الهندسية
            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3, google_api_key=MY_SECRET_KEY)

            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({
                "context": context,
                "history": self.chat_history,
                "question": user_query
            })

            self.chat_history.append(HumanMessage(content=user_query))
            self.chat_history.append(AIMessage(content=response))
            
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]

            self.root.after(0, lambda: self._safe_ui_update(response))
            
        except Exception as e:
            self.root.after(0, lambda: self.append_to_display(f"❌ خطأ أثناء توليد الصياغة: {e}", "system_alert"))
            self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))

    def _safe_ui_update(self, response):
        self.append_to_display(response, "bot_msg")
        self.send_btn.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = ElectricChatApp(root)
    root.mainloop()
