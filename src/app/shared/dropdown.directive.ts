import { Directive, ElementRef, HostBinding, HostListener, Renderer2 } from '@angular/core';

@Directive({
  selector: '[appDropdown]'
})
export class DropdownDirective {
  @HostBinding('class.show') isOpen = false;

  constructor(private el: ElementRef, private renderer: Renderer2) { }
  @HostListener('document:click', ['$event']) toggleOpen(event: Event) {
    this.isOpen = this.el.nativeElement.contains(event.target) ? !this.isOpen : false;
    let modal = this.el.nativeElement.querySelector('div');
    if (this.isOpen) {
      this.renderer.addClass(modal, 'show');
    } else {
      this.renderer.removeClass(modal, 'show');
    }
  }
}