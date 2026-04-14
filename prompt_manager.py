import os
from datetime import datetime  # Add this import
from datetime import timezone
from typing import Dict, List, Optional
from models import db, PromptConfig, PromptComponent

class PromptManager:
    def __init__(self):
        # No in-memory defaults: prompt content lives in the database.
        pass
    
    def clean_uncategorized_instruction(self, text: str) -> str:
        """Remove explicit instructions that encourage 'Uncategorized' output."""
        if not text:
            return text

        lines = text.splitlines()
        cleaned = []
        for line in lines:
            lower = line.lower().strip()
            if 'uncategorized' in lower and ('jawab' in lower or 'jika teks' in lower or 'jika tidak cocok' in lower):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned).strip()

    def initialize_default_config(self):
        """Ensure there is an active prompt configuration in the database."""
        active_config = PromptConfig.query.filter_by(is_active=True).first()
        if active_config:
            return

        first_config = PromptConfig.query.order_by(PromptConfig.id.asc()).first()
        if not first_config:
            raise RuntimeError(
                'No prompt configuration found in the database. Create one in Prompt Manager first.'
            )

        first_config.is_active = True
        db.session.commit()
    
    def get_active_config(self) -> Optional[PromptConfig]:
        """Get currently active prompt configuration"""
        return PromptConfig.query.filter_by(is_active=True).first()
    
    def get_config_components(self, config_id: int) -> List[PromptComponent]:
        """Get all components for a configuration"""
        return PromptComponent.query.filter_by(
            config_id=config_id
        ).order_by('order_index').all()
    
    def build_prompt(self, text: str, config_id: Optional[int] = None) -> str:
        """Build prompt from database configuration"""
        if not config_id:
            config = self.get_active_config()
            if not config:
                self.initialize_default_config()
                config = self.get_active_config()
            config_id = config.id
        
        components = self.get_config_components(config_id)
        base_component = next((c for c in components if c.component_type == 'base'), None)

        if not base_component:
            raise RuntimeError(f'Prompt config {config_id} is missing the base component')

        base_text = self.clean_uncategorized_instruction(
            base_component.content.replace('{text}', text).rstrip()
        )
        extra_components = []
        for comp in components:
            if comp.component_type != 'base' and comp.is_enabled:
                extra_components.append(self.clean_uncategorized_instruction(comp.content.strip()))

        extra_components = [c for c in extra_components if c]

        if not extra_components:
            return base_text

        if '<|im_start|>assistant' in base_text and '<|im_end|>' not in base_text:
            # Base prompt already opens the assistant block; keep additional
            # instructions inside the same ChatML assistant message.
            return base_text + '\n\n' + '\n\n'.join(extra_components)

        return '\n\n'.join([base_text] + extra_components)
    
    def create_new_config(self, name: str, description: str = "", copy_from: Optional[int] = None) -> PromptConfig:
        """Create new prompt configuration"""
        new_config = PromptConfig(
            name=name,
            description=description,
            is_active=False
        )
        db.session.add(new_config)
        db.session.flush()
        
        # Copy components from existing config or use defaults
        if copy_from:
            source_components = self.get_config_components(copy_from)
        else:
            source_config = self.get_active_config() or PromptConfig.query.order_by(PromptConfig.id.asc()).first()
            if not source_config:
                raise RuntimeError('No source prompt configuration available to copy from.')
            source_components = self.get_config_components(source_config.id)
        
        for source_comp in source_components:
            new_component = PromptComponent(
                config_id=new_config.id,
                component_type=source_comp.component_type,
                content=source_comp.content,
                is_enabled=source_comp.is_enabled,
                order_index=source_comp.order_index
            )
            db.session.add(new_component)
        
        db.session.commit()
        return new_config
    
    def activate_config(self, config_id: int):
        """Activate a specific configuration"""
        # Deactivate all others
        PromptConfig.query.update({PromptConfig.is_active: False})
        
        # Activate selected one
        config = PromptConfig.query.get(config_id)
        if config:
            config.is_active = True
            db.session.commit()
    
    def update_component(self, component_id: int, content: str = None, is_enabled: bool = None):
        """Update component content and status"""
        try:
            component = PromptComponent.query.get(component_id)
            if component:
                if content is not None:
                    component.content = content
                if is_enabled is not None:
                    component.is_enabled = is_enabled
                component.updated_at = datetime.now(timezone.utc)
                db.session.commit()
                return True
            return False
        except Exception as e:
            db.session.rollback()
            print(f"Error in update_component: {str(e)}")
            raise e

# Global prompt manager instance
prompt_manager = PromptManager()