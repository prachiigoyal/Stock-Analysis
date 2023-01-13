import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { TypewriterModule } from '@typewriterjs/typewriterjs-angular';
import { IvyCarouselModule } from 'angular-responsive-carousel';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { HeaderComponent } from './components/header/header.component';
import { DropdownDirective } from './shared/dropdown.directive';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { PageHomeComponent } from './pages/page-home/page-home.component';
import { BannerHeadComponent } from './components/banner-head/banner-head.component';
import { FeatureCardComponent } from './components/feature-card/feature-card.component';
import { PageProductComponent } from './pages/page-product/page-product.component';
import { PageCategoryComponent } from './pages/page-category/page-category.component';
import { FooterComponent } from './components/footer/footer.component';
import { PageBlogComponent } from './pages/page-blog/page-blog.component';
import { PageNotFoundComponent } from './pages/page-not-found/page-not-found.component';
import { PagePrivacyComponent } from './pages/page-privacy/page-privacy.component';
import { PageAgreementComponent } from './pages/page-agreement/page-agreement.component';
import { PageFaqsComponent } from './pages/page-faqs/page-faqs.component';
import { FaqComponent } from './components/faq/faq.component';
import { SpinnerComponent } from './components/spinner/spinner.component';
import { TransDirective } from './shared/trans.directive';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    DropdownDirective,
    PageHomeComponent,
    BannerHeadComponent,
    FeatureCardComponent,
    PageProductComponent,
    PageCategoryComponent,
    FooterComponent,
    PageBlogComponent,
    PageNotFoundComponent,
    PagePrivacyComponent,
    PageAgreementComponent,
    PageFaqsComponent,
    FaqComponent,
    SpinnerComponent,
    TransDirective,
  ],
  imports: [
    AppRoutingModule,
    BrowserModule,
    BrowserAnimationsModule,
    TypewriterModule,
    IvyCarouselModule,
  ],
  providers: [],
  bootstrap: [AppComponent],
})
export class AppModule {}
