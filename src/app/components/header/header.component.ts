import {
  animate,
  state,
  style,
  transition,
  trigger,
} from '@angular/animations';
import { Component, HostListener, OnDestroy, OnInit } from '@angular/core';
import { timer } from 'rxjs';
import * as submenuStructureData from '../../data/submenu-structure.json';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  animations: [
    trigger('submenuState', [
      state(
        'normal',
        style({
          opacity: '0',
        })
      ),
      state(
        'focus',
        style({
          opacity: '1',
        })
      ),
      transition('normal <=> focus', animate(150)),
      transition('focus => normal', animate(150)),
    ]),
  ],
  styleUrls: ['./header.component.css'],
})
export class HeaderComponent implements OnInit, OnDestroy {
  private leaveObservable: any;
  private clickedInside = false;

  submenuState = 'normal';
  currentSubMenu = 0;
  currentMobileSubMenu = 0;
  isMobileMenuOpened = false;
  pageYOffset = 0;
  submenuStructure = (submenuStructureData as any).default;

  constructor() {}
  ngOnInit(): void {}
  onMobileMenuClick() {
    this.isMobileMenuOpened = !this.isMobileMenuOpened;
    this.currentMobileSubMenu = 0;
  }
  @HostListener('window:scroll', ['$event']) onScrollEvent($event) {
    this.pageYOffset = window.pageYOffset;
  }
  onMouseEnterLink(hoverAt: number) {
    if (this.leaveObservable) {
      this.leaveObservable.unsubscribe();
    }
    if (hoverAt >= 0) {
      this.isMobileMenuOpened = false;
      this.currentSubMenu = hoverAt;
      this.submenuState = 'focus';
    }
  }
  onMouseLeaveLink() {
    if (this.submenuState !== 'focus') {
      return;
    }

    this.leaveObservable = timer(100).subscribe(() => {
      this.submenuState = 'normal';
    });
  }
  onMobileMenuOptionClick(i: number) {
    this.currentMobileSubMenu = i;
  }
  @HostListener('click')
  clicked() {
    this.clickedInside = true;
  }
  @HostListener('document:click')
  clickedOut() {
    if (!this.clickedInside) {
      this.isMobileMenuOpened = false;
    }
    this.clickedInside = false;
  }
  ngOnDestroy(): void {
    if (this.leaveObservable) {
      this.leaveObservable.unsubscribe();
    }
  }
}
