import os
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, MessageHandler, CommandHandler, CallbackQueryHandler, filters, ContextTypes
import openai
from pydub import AudioSegment
import tempfile
import asyncio

# Environment variables
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
YOUR_TELEGRAM_ID = int(os.environ.get("YOUR_TELEGRAM_ID"))

# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# User preferences (stored in memory, could be moved to a JSON file)
USER_SETTINGS = {
    "model": "gpt-4o-mini",  # Default model
    "format": "summary",      # Default format
    "language": "same",        # Default language (same as input)
    "auto_transcribe": True,   # Auto transcribe without asking
    "include_transcript": True, # Include transcript with summary
    "max_length": "medium"     # Summary length
}

# Format templates
FORMAT_TEMPLATES = {
    "summary": "Provide a concise summary of this voice note in 2-3 sentences:",
    "bullets": "Summarize this voice note in clear bullet points:",
    "keypoints": "Extract the key points from this voice note (numbered list):",
    "action": "List any action items or tasks mentioned in this voice note:",
    "meeting": "Format this as meeting notes with: Participants (if mentioned), Topics discussed, Decisions made, Action items:",
    "idea": "Extract and structure the main idea, supporting points, and any questions or concerns:",
    "todo": "Extract all TODO items, deadlines, and commitments mentioned:",
    "technical": "Provide a technical summary focusing on: Problem, Solution, Implementation details, Next steps:",
    "analysis": "Analyze this content and provide: Main argument, Supporting evidence, Counterpoints (if any), Conclusion:"
}

# Available models
MODELS = {
    "gpt-4o-mini": "Fast & Cheap",
    "gpt-4o": "Most Capable",
    "gpt-3.5-turbo": "Legacy Fast"
}

class VoiceBot:
    def __init__(self):
        self.processing_messages = {}
        
    async def check_user(self, update: Update) -> bool:
        """Security check - only respond to owner"""
        user_id = update.effective_user.id if update.effective_user else None
        return user_id == YOUR_TELEGRAM_ID
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_user(update):
            return
        
        welcome_text = """
🎙️ **Personal Voice Assistant Ready!**

Send me a voice note and I'll transcribe and format it for you.

**Quick Commands:**
• /settings - Configure preferences
• /format - Choose output format
• /model - Switch AI model
• /stats - View usage statistics
• /help - Show all commands

**Quick Format Commands:**
• /summary - Summarize last transcription
• /bullets - Convert to bullet points
• /action - Extract action items
• /todo - Extract TODOs

Current settings:
• Model: {model}
• Format: {format}
• Auto-transcribe: {auto}
        """.format(
            model=USER_SETTINGS["model"],
            format=USER_SETTINGS["format"],
            auto="✅" if USER_SETTINGS["auto_transcribe"] else "❌"
        )
        
        await update.message.reply_text(welcome_text, parse_mode="Markdown")
    
    async def settings_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_user(update):
            return
        
        keyboard = [
            [InlineKeyboardButton("📝 Output Format", callback_data="setting_format")],
            [InlineKeyboardButton("🤖 AI Model", callback_data="setting_model")],
            [InlineKeyboardButton("🌍 Language", callback_data="setting_language")],
            [InlineKeyboardButton("📊 Summary Length", callback_data="setting_length")],
            [
                InlineKeyboardButton(
                    f"{'✅' if USER_SETTINGS['auto_transcribe'] else '❌'} Auto-transcribe",
                    callback_data="toggle_auto"
                ),
                InlineKeyboardButton(
                    f"{'✅' if USER_SETTINGS['include_transcript'] else '❌'} Include transcript",
                    callback_data="toggle_transcript"
                )
            ],
            [InlineKeyboardButton("📈 View Stats", callback_data="view_stats")],
            [InlineKeyboardButton("❌ Close", callback_data="close")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        settings_text = f"""
⚙️ **Current Settings**

**Model:** `{USER_SETTINGS['model']}` ({MODELS.get(USER_SETTINGS['model'], 'Custom')})
**Format:** {USER_SETTINGS['format']}
**Language:** {USER_SETTINGS['language']}
**Summary Length:** {USER_SETTINGS['max_length']}
**Auto-transcribe:** {'✅ Enabled' if USER_SETTINGS['auto_transcribe'] else '❌ Disabled'}
**Include Transcript:** {'✅ Yes' if USER_SETTINGS['include_transcript'] else '❌ No'}
        """
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                settings_text, 
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                settings_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
    
    async def handle_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_user(update):
            return
        
        # Store voice file ID for potential reprocessing
        context.user_data['last_voice_id'] = update.message.voice.file_id
        context.user_data['last_voice_duration'] = update.message.voice.duration
        
        if not USER_SETTINGS["auto_transcribe"]:
            keyboard = [
                [
                    InlineKeyboardButton("📝 Transcribe", callback_data="transcribe_now"),
                    InlineKeyboardButton("🎯 Quick Format", callback_data="quick_format")
                ],
                [InlineKeyboardButton("❌ Skip", callback_data="close")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text(
                f"🎤 Voice note received ({update.message.voice.duration}s)\nWhat would you like to do?",
                reply_markup=reply_markup
            )
        else:
            await self.process_voice(update, context)
    
    async def process_voice(self, update: Update, context: ContextTypes.DEFAULT_TYPE, custom_format=None):
        # Send processing message
        if update.callback_query:
            status = await update.callback_query.message.reply_text("🎤 Transcribing...")
            await update.callback_query.answer()
        else:
            status = await update.message.reply_text("🎤 Transcribing...")
        
        self.processing_messages[update.effective_user.id] = status.message_id
        
        try:
            # Get voice file
            voice_file_id = context.user_data.get('last_voice_id')
            if not voice_file_id and update.message and update.message.voice:
                voice_file_id = update.message.voice.file_id
            
            voice_file = await context.bot.get_file(voice_file_id)
            
            # Download and convert
            with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
                await voice_file.download_to_drive(tmp_file.name)
                
                # Convert to mp3
                audio = AudioSegment.from_ogg(tmp_file.name)
                mp3_path = tmp_file.name.replace('.ogg', '.mp3')
                audio.export(mp3_path, format="mp3")
                
                # Transcribe
                with open(mp3_path, "rb") as audio_file:
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="text"
                    )
                
                # Store transcript for reprocessing
                context.user_data['last_transcript'] = transcript
                context.user_data['last_transcript_time'] = datetime.now().isoformat()
                
                # Update status
                await status.edit_text("📝 Formatting...")
                
                # Get format template
                format_type = custom_format or USER_SETTINGS["format"]
                format_prompt = FORMAT_TEMPLATES.get(format_type, FORMAT_TEMPLATES["summary"])
                
                # Add language instruction if needed
                language_instruction = ""
                if USER_SETTINGS["language"] != "same":
                    language_instruction = f"\nRespond in {USER_SETTINGS['language']}."
                
                # Add length instruction
                length_instruction = {
                    "short": "\nKeep it very brief (1-2 sentences or 3-4 bullet points).",
                    "medium": "\nProvide a moderate level of detail.",
                    "long": "\nProvide comprehensive detail with all important points.",
                    "full": "\nProvide maximum detail, missing nothing important."
                }.get(USER_SETTINGS["max_length"], "")
                
                # Create prompt
                full_prompt = f"{format_prompt}{language_instruction}{length_instruction}\n\nTranscription:\n{transcript}"
                
                # Get AI response
                response = openai.chat.completions.create(
                    model=USER_SETTINGS["model"],
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that processes voice note transcriptions according to specific formatting requirements."},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                formatted_text = response.choices[0].message.content
                
                # Store formatted result
                context.user_data['last_formatted'] = formatted_text
                context.user_data['last_format_type'] = format_type
                
                # Prepare output
                output = f"**📋 {format_type.upper()}**\n\n{formatted_text}"
                
                if USER_SETTINGS["include_transcript"]:
                    output += f"\n\n**📝 Original Transcript:**\n_{transcript}_"
                
                # Add action buttons
                keyboard = [
                    [
                        InlineKeyboardButton("🔄 Reformat", callback_data="show_formats"),
                        InlineKeyboardButton("📊 Different Model", callback_data="change_model_once")
                    ],
                    [
                        InlineKeyboardButton("📝 Show Transcript", callback_data="show_transcript") 
                        if not USER_SETTINGS["include_transcript"] else
                        InlineKeyboardButton("📝 Hide Transcript", callback_data="hide_transcript")
                    ],
                    [InlineKeyboardButton("✅ Done", callback_data="close")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Send results
                await status.edit_text(
                    output[:4000],  # Telegram message limit
                    parse_mode="Markdown",
                    reply_markup=reply_markup
                )
                
                # Update stats
                if 'stats' not in context.user_data:
                    context.user_data['stats'] = {
                        'total_transcriptions': 0,
                        'total_duration': 0,
                        'formats_used': {},
                        'models_used': {}
                    }
                
                stats = context.user_data['stats']
                stats['total_transcriptions'] += 1
                stats['total_duration'] += context.user_data.get('last_voice_duration', 0)
                stats['formats_used'][format_type] = stats['formats_used'].get(format_type, 0) + 1
                stats['models_used'][USER_SETTINGS["model"]] = stats['models_used'].get(USER_SETTINGS["model"], 0) + 1
                
                # Clean up temp files
                os.unlink(tmp_file.name)
                os.unlink(mp3_path)
                
        except Exception as e:
            await status.edit_text(f"❌ Error: {str(e)}")
    
    async def show_formats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show format options for reprocessing"""
        keyboard = []
        for format_key, description in FORMAT_TEMPLATES.items():
            keyboard.append([InlineKeyboardButton(
                f"{'✓ ' if format_key == USER_SETTINGS['format'] else ''}{format_key.title()}",
                callback_data=f"reformat_{format_key}"
            )])
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="close")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            "Choose a format for your transcription:",
            reply_markup=reply_markup
        )
        await update.callback_query.answer()
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not await self.check_user(update):
            return
        
        query = update.callback_query
        data = query.data
        
        if data == "close":
            await query.message.delete()
            await query.answer("Closed")
            
        elif data == "transcribe_now":
            await self.process_voice(update, context)
            
        elif data.startswith("reformat_"):
            format_type = data.replace("reformat_", "")
            await query.answer(f"Reformatting as {format_type}...")
            await self.reprocess_with_format(update, context, format_type)
            
        elif data == "setting_format":
            await self.show_format_settings(update, context)
            
        elif data == "setting_model":
            await self.show_model_settings(update, context)
            
        elif data == "setting_language":
            await self.show_language_settings(update, context)
            
        elif data == "setting_length":
            await self.show_length_settings(update, context)
            
        elif data.startswith("set_format_"):
            new_format = data.replace("set_format_", "")
            USER_SETTINGS["format"] = new_format
            await query.answer(f"Default format set to: {new_format}")
            await self.settings_menu(update, context)
            
        elif data.startswith("set_model_"):
            new_model = data.replace("set_model_", "")
            USER_SETTINGS["model"] = new_model
            await query.answer(f"Model changed to: {new_model}")
            await self.settings_menu(update, context)
            
        elif data.startswith("set_language_"):
            new_language = data.replace("set_language_", "")
            USER_SETTINGS["language"] = new_language
            await query.answer(f"Language set to: {new_language}")
            await self.settings_menu(update, context)
            
        elif data.startswith("set_length_"):
            new_length = data.replace("set_length_", "")
            USER_SETTINGS["max_length"] = new_length
            await query.answer(f"Summary length set to: {new_length}")
            await self.settings_menu(update, context)
            
        elif data == "toggle_auto":
            USER_SETTINGS["auto_transcribe"] = not USER_SETTINGS["auto_transcribe"]
            await query.answer(f"Auto-transcribe {'enabled' if USER_SETTINGS['auto_transcribe'] else 'disabled'}")
            await self.settings_menu(update, context)
            
        elif data == "toggle_transcript":
            USER_SETTINGS["include_transcript"] = not USER_SETTINGS["include_transcript"]
            await query.answer(f"Transcript inclusion {'enabled' if USER_SETTINGS['include_transcript'] else 'disabled'}")
            await self.settings_menu(update, context)
            
        elif data == "view_stats":
            await self.show_stats(update, context)
            
        elif data == "show_transcript":
            transcript = context.user_data.get('last_transcript', 'No transcript available')
            await query.message.reply_text(f"**📝 Original Transcript:**\n\n{transcript}", parse_mode="Markdown")
            await query.answer("Showing transcript")
            
        elif data == "change_model_once":
            await self.show_model_quick_change(update, context)
    
    async def show_format_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = []
        for format_key in FORMAT_TEMPLATES.keys():
            keyboard.append([InlineKeyboardButton(
                f"{'✓ ' if format_key == USER_SETTINGS['format'] else ''}{format_key.title()}",
                callback_data=f"set_format_{format_key}"
            )])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_settings")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            "📝 **Choose Default Format:**\n\nThis will be used for all voice notes unless you specify otherwise.",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def show_model_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = []
        for model_key, description in MODELS.items():
            keyboard.append([InlineKeyboardButton(
                f"{'✓ ' if model_key == USER_SETTINGS['model'] else ''}{model_key} - {description}",
                callback_data=f"set_model_{model_key}"
            )])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_settings")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            "🤖 **Choose AI Model:**\n\n• `gpt-4o-mini`: Fast, cheap, great for most tasks\n• `gpt-4o`: Most capable, best quality\n• `gpt-3.5-turbo`: Legacy, fastest",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def show_language_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        languages = {
            "same": "Same as input",
            "English": "English",
            "Spanish": "Spanish",
            "French": "French",
            "German": "German",
            "Italian": "Italian",
            "Portuguese": "Portuguese",
            "Chinese": "Chinese",
            "Japanese": "Japanese",
            "Korean": "Korean"
        }
        
        keyboard = []
        for lang_key, lang_name in languages.items():
            keyboard.append([InlineKeyboardButton(
                f"{'✓ ' if lang_key == USER_SETTINGS['language'] else ''}{lang_name}",
                callback_data=f"set_language_{lang_key}"
            )])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_settings")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            "🌍 **Choose Output Language:**",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def show_length_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        lengths = {
            "short": "Short (1-2 sentences)",
            "medium": "Medium (balanced)",
            "long": "Long (detailed)",
            "full": "Full (comprehensive)"
        }
        
        keyboard = []
        for length_key, description in lengths.items():
            keyboard.append([InlineKeyboardButton(
                f"{'✓ ' if length_key == USER_SETTINGS['max_length'] else ''}{description}",
                callback_data=f"set_length_{length_key}"
            )])
        keyboard.append([InlineKeyboardButton("⬅️ Back", callback_data="back_to_settings")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            "📊 **Choose Summary Length:**",
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        stats = context.user_data.get('stats', {
            'total_transcriptions': 0,
            'total_duration': 0,
            'formats_used': {},
            'models_used': {}
        })
        
        stats_text = f"""
📈 **Usage Statistics**

**Total Transcriptions:** {stats['total_transcriptions']}
**Total Duration:** {stats['total_duration']}s (~{stats['total_duration']//60}min)

**Formats Used:**
"""
        for format_type, count in stats.get('formats_used', {}).items():
            stats_text += f"• {format_type}: {count} times\n"
        
        stats_text += "\n**Models Used:**\n"
        for model, count in stats.get('models_used', {}).items():
            stats_text += f"• {model}: {count} times\n"
        
        keyboard = [[InlineKeyboardButton("⬅️ Back", callback_data="back_to_settings")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            stats_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
    
    async def reprocess_with_format(self, update: Update, context: ContextTypes.DEFAULT_TYPE, format_type: str):
        """Reprocess last transcript with new format"""
        transcript = context.user_data.get('last_transcript')
        
        if not transcript:
            await update.callback_query.answer("No transcript available to reformat")
            return
        
        try:
            # Get format template
            format_prompt = FORMAT_TEMPLATES.get(format_type, FORMAT_TEMPLATES["summary"])
            
            # Add language and length instructions
            language_instruction = ""
            if USER_SETTINGS["language"] != "same":
                language_instruction = f"\nRespond in {USER_SETTINGS['language']}."
            
            length_instruction = {
                "short": "\nKeep it very brief.",
                "medium": "\nProvide moderate detail.",
                "long": "\nProvide comprehensive detail.",
                "full": "\nProvide maximum detail."
            }.get(USER_SETTINGS["max_length"], "")
            
            full_prompt = f"{format_prompt}{language_instruction}{length_instruction}\n\nTranscription:\n{transcript}"
            
            # Get AI response
            response = openai.chat.completions.create(
                model=USER_SETTINGS["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that processes voice note transcriptions."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7
            )
            
            formatted_text = response.choices[0].message.content
            
            # Prepare output
            output = f"**📋 {format_type.upper()}**\n\n{formatted_text}"
            
            if USER_SETTINGS["include_transcript"]:
                output += f"\n\n**📝 Original Transcript:**\n_{transcript}_"
            
            # Add action buttons
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Try Another Format", callback_data="show_formats"),
                    InlineKeyboardButton("✅ Done", callback_data="close")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.callback_query.edit_message_text(
                output[:4000],
                parse_mode="Markdown",
                reply_markup=reply_markup
            )
            
        except Exception as e:
            await update.callback_query.answer(f"Error: {str(e)}")
    
    async def show_model_quick_change(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick model change for reprocessing"""
        keyboard = []
        for model_key, description in MODELS.items():
            if model_key != USER_SETTINGS["model"]:
                keyboard.append([InlineKeyboardButton(
                    f"{model_key} - {description}",
                    callback_data=f"quick_model_{model_key}"
                )])
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="close")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            "Choose a model for one-time reprocessing:",
            reply_markup=reply_markup
        )
    
    async def quick_format_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE, format_type: str):
        """Handle quick format commands like /bullets, /summary, etc."""
        if not await self.check_user(update):
            return
        
        transcript = context.user_data.get('last_transcript')
        if not transcript:
            await update.message.reply_text("❌ No recent transcription to format. Send a voice note first!")
            return
        
        # Process with specified format
        await update.message.reply_text(f"Reformatting as {format_type}...")
        
        try:
            format_prompt = FORMAT_TEMPLATES.get(format_type, FORMAT_TEMPLATES["summary"])
            
            response = openai.chat.completions.create(
                model=USER_SETTINGS["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{format_prompt}\n\n{transcript}"}
                ]
            )
            
            formatted_text = response.choices[0].message.content
            output = f"**📋 {format_type.upper()}**\n\n{formatted_text}"
            
            await update.message.reply_text(output, parse_mode="Markdown")
            
        except Exception as e:
            await update.message.reply_text(f"❌ Error: {str(e)}")

# Initialize bot instance
bot = VoiceBot()

def main():
    """Main function to run the bot"""
    # Create application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Command handlers
    app.add_handler(CommandHandler("start", bot.start))
    app.add_handler(CommandHandler("help", bot.start))
    app.add_handler(CommandHandler("settings", bot.settings_menu))
    
    # Quick format commands
    app.add_handler(CommandHandler("summary", lambda u, c: bot.quick_format_command(u, c, "summary")))
    app.add_handler(CommandHandler("bullets", lambda u, c: bot.quick_format_command(u, c, "bullets")))
    app.add_handler(CommandHandler("action", lambda u, c: bot.quick_format_command(u, c, "action")))
    app.add_handler(CommandHandler("todo", lambda u, c: bot.quick_format_command(u, c, "todo")))
    app.add_handler(CommandHandler("keypoints", lambda u, c: bot.quick_format_command(u, c, "keypoints")))
    app.add_handler(CommandHandler("meeting", lambda u, c: bot.quick_format_command(u, c, "meeting")))
    
    # Message handlers
    app.add_handler(MessageHandler(filters.VOICE, bot.handle_voice))
    
    # Callback query handler
    app.add_handler(CallbackQueryHandler(bot.handle_callback))
    
    # Start polling
    print(f"🤖 Bot started! Only responding to user ID: {YOUR_TELEGRAM_ID}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
